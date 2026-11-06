"""Dataset/sampler module (gameplan module 6): wires the manifest, the
assembled label table (module "final_labels"), and the volume loader/cache
into one `torch.utils.data.Dataset`.

Scope decisions, and why:

- **`KneeStudyDataset` only ever reads the precomputed volume cache
  (`scripts/precompute_volume_cache.py`'s output), never a raw .dcm.** This is
  the [redacted: private project notes] working agreement ("no training run should ever touch raw
  DICOM") made structural rather than conventional: a missing cache file is a
  clear `FileNotFoundError` naming the precompute script, not a silent
  fall-through to on-the-fly DICOM decoding.
- **A study's variable series count (3-14, avg 5.5) collapses to one series
  per canonical plane** (Sagittal/Coronal/Axial), chosen by `select_canonical_
  series` as the series with the most slices in that plane. This is a
  reasonable first default, not a claim that it's the right one for a model
  that doesn't exist yet (module 7) -- e.g. it ignores Fluid_Sensitive/
  Fat_Suppression sequence-type information entirely. A study missing a plane
  contributes a zero volume for it, flagged in `plane_present` rather than
  silently treated as real data.
- **The label array's MASKED (-1) cells are returned as-is, with an explicit
  boolean mask**, not dropped or imputed -- the training loop (module 8)
  decides how to use them (e.g. excluded from the loss, or fed to a
  PU-learning-style objective per [redacted: private project notes]'s "optimize the score" priority
  for labels like Synovitis that are positive-only).
"""

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from data.volume_loader import center_crop_or_pad
from data.label_mining.final_labels import MASKED
from utils.constants import LABEL_COLUMNS

PLANES = ("Sagittal", "Coronal", "Axial")


def spacing_tag(target_spacing) -> str:
    """Filesystem-safe tag for a (dz, dy, dx) spacing tuple, used to keep
    volume caches computed at different spacings from colliding -- see
    scripts/precompute_volume_cache.py.
    """
    return "x".join(f"{v:g}" for v in target_spacing)


def select_canonical_series(series_rows, planes=PLANES, plane_key="csv_Anatomical_Plane",
                             n_slices_key="n_slices", series_id_key="SeriesInstanceUID"):
    """From a study's series manifest rows, pick one series per plane: the one
    with the most slices (a simple, documented default -- see module
    docstring). `series_rows`: list of dict-like rows (e.g. from
    series_manifest.csv, pre-filtered to one study). Returns
    {plane: series_id}; a plane absent from `series_rows` is absent from the
    result, not filled with a guess.
    """
    best = {}
    for row in series_rows:
        plane = row.get(plane_key)
        if plane not in planes:
            continue
        n = int(row[n_slices_key])
        if plane not in best or n > best[plane][0]:
            best[plane] = (n, row[series_id_key])
    return {plane: series_id for plane, (_, series_id) in best.items()}


class VolumeCacheMissingError(FileNotFoundError):
    def __init__(self, path):
        super().__init__(
            f"{path} not found. KneeStudyDataset never reads raw DICOM directly -- "
            f"run scripts/precompute_volume_cache.py first to populate the volume cache "
            f"(see that script's docstring for the local-vs-full-dataset caveat)."
        )


def load_cached_volume(cache_dir, study_id, series_id):
    """(D, H, W) float32 array from the precomputed cache written by
    scripts/precompute_volume_cache.py. Raises VolumeCacheMissingError, not a
    generic FileNotFoundError, so the fix is obvious from the error alone.
    """
    path = Path(cache_dir) / study_id / f"{series_id}.npy"
    if not path.exists():
        raise VolumeCacheMissingError(path)
    return np.load(path)


class KneeStudyDataset(Dataset):
    """One item = one study: a (len(planes), *target_shape) image tensor (one
    channel per canonical plane, zero-filled where the study has no series for
    that plane), the study's 12-column label vector, a mask of which labels
    are actually known (not MASKED), and a mask of which planes are real vs.
    zero-filled.
    """

    def __init__(self, study_ids, labels, series_by_study, cache_dir,
                 target_shape=(32, 256, 256), planes=PLANES, columns=LABEL_COLUMNS, weights=None):
        labels = np.asarray(labels)
        if labels.shape != (len(study_ids), len(columns)):
            raise ValueError(
                f"labels shape {labels.shape} != (n_studies={len(study_ids)}, "
                f"n_columns={len(columns)})"
            )
        self.study_ids = list(study_ids)
        self.labels = labels
        # per-cell loss weights (final_labels.label_weights); all 1.0 if not given
        self.weights = (np.ones(labels.shape, dtype=np.float32) if weights is None
                        else np.asarray(weights, dtype=np.float32))
        self.series_by_study = series_by_study
        self.cache_dir = Path(cache_dir)
        self.target_shape = target_shape
        self.planes = planes
        self.columns = columns

    def __len__(self):
        return len(self.study_ids)

    def __getitem__(self, idx):
        study_id = self.study_ids[idx]
        canonical = select_canonical_series(self.series_by_study.get(study_id, []), self.planes)

        channels = []
        plane_present = []
        for plane in self.planes:
            series_id = canonical.get(plane)
            if series_id is None:
                channels.append(np.zeros(self.target_shape, dtype=np.float32))
                plane_present.append(False)
            else:
                volume = load_cached_volume(self.cache_dir, study_id, series_id)
                channels.append(center_crop_or_pad(volume, self.target_shape))
                plane_present.append(True)
        image = np.stack(channels, axis=0)  # (n_planes, D, H, W)

        label = self.labels[idx].astype(np.float32)
        label_mask = label != MASKED

        return {
            "study_id": study_id,
            "image": torch.from_numpy(image),
            "label": torch.from_numpy(label),
            "label_mask": torch.from_numpy(label_mask),
            "label_weight": torch.from_numpy(self.weights[idx]),
            "plane_present": torch.tensor(plane_present, dtype=torch.bool),
        }
