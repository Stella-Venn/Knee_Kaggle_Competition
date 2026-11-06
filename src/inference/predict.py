"""Inference: test-set DICOMs -> submission.csv (module 9, 2026-09-24).

Preprocessing is the SAME chain the training cache used (volume-cache kernel,
scripts/build_volume_cache_kernel.py): canonical series per plane (most
slices), geometry-sorted load, resample to TARGET_SPACING mm, per-volume
percentile normalization, center crop/pad to TARGET_SHAPE. A mismatch here
would silently feed the model inputs unlike anything it trained on, so these
constants must match the cache the deployed weights were trained on.

[redacted: detail about the data challenge this code was written for]
missing plane (the model already handles those, KneeMultiViewModel masks
them), and a study with no loadable plane at all gets FALLBACK_PROB in every
column (0.5 is rank-neutral for AUC). Every study in test.csv always gets
exactly one row.
"""

import csv
import os
import time
from multiprocessing import get_context

import numpy as np
import torch

from data.dataset import select_canonical_series
from data.volume_loader import (
    center_crop_or_pad,
    load_series_volume,
    normalize_intensity,
    resample_volume,
)
from utils.constants import LABEL_COLUMNS

PLANES = ("Sagittal", "Coronal", "Axial")
TARGET_SPACING = (3.0, 0.5, 0.5)  # must match the volume cache the weights were trained on
TARGET_SHAPE = (32, 256, 256)
FALLBACK_PROB = 0.5
ID_COLUMN = "StudyInstanceUID"


def _n_dcm_files(path):
    try:
        return sum(1 for f in os.listdir(path) if f.endswith(".dcm"))
    except FileNotFoundError:
        return 0


def read_series_csv(path):
    """{study_id: [row dicts]} from test_series.csv / train_series.csv."""
    by_study = {}
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            by_study.setdefault(row[ID_COLUMN], []).append(row)
    return by_study


def preprocess_study(study_dir, series_rows, planes=PLANES,
                     target_spacing=TARGET_SPACING, target_shape=TARGET_SHAPE):
    """(image (P, D, H, W) float32, plane_present list[bool], errors list[str]).

    Same canonical-series choice as the cache kernel: among series whose
    directory actually has .dcm files, the one with the most slices per CSV
    plane. A plane with no series, or whose series fails to load, is a
    zero placeholder with plane_present=False -- recorded in `errors`,
    never raised.
    """
    candidates = []
    for r in series_rows:
        n = _n_dcm_files(os.path.join(study_dir, r["SeriesInstanceUID"]))
        if n:
            candidates.append({"SeriesInstanceUID": r["SeriesInstanceUID"], "n_slices": n,
                               "csv_Anatomical_Plane": r["Anatomical_Plane"]})
    canonical = select_canonical_series(candidates, planes=planes)

    channels, present, errors = [], [], []
    for plane in planes:
        series_id = canonical.get(plane)
        volume = None
        if series_id is not None:
            try:
                sv = load_series_volume(os.path.join(study_dir, series_id))
                resampled, _ = resample_volume(
                    sv.volume, (sv.slice_spacing, sv.row_spacing, sv.col_spacing), target_spacing)
                volume = center_crop_or_pad(normalize_intensity(resampled), target_shape)
            except Exception as e:  # noqa: BLE001 -- one bad series must not kill the run
                errors.append(f"{plane} {series_id}: {e!r}")
        else:
            errors.append(f"{plane}: no series")
        present.append(volume is not None)
        channels.append(volume if volume is not None else np.zeros(target_shape, dtype=np.float32))
    return np.stack(channels).astype(np.float32), present, errors


def _prep_task(args):
    """Worker-safe preprocess_study: never raises (a study-level failure comes
    back as no image + an error), so one bad study can't kill the pool."""
    study_dir, series_rows = args
    try:
        return preprocess_study(study_dir, series_rows)
    except Exception as e:  # noqa: BLE001
        return None, [False] * len(PLANES), [f"study-level: {e!r}"]


@torch.no_grad()
def predict_study(model, image, present, device="cpu"):
    """(n_classes,) float probabilities for one preprocessed study."""
    x = torch.from_numpy(image)[None].to(device)
    p = torch.tensor([present], dtype=torch.bool, device=device)
    return torch.sigmoid(model(x, p)).float().cpu().numpy()[0]


def run_inference(comp_dir, model, out_path, device="cpu", columns=LABEL_COLUMNS, log=print,
                  n_workers=1):
    """Predict every study in comp_dir/test.csv and write out_path. Returns a
    summary dict (counts, per-study seconds, errors) for logging.

    n_workers > 1 preprocesses in that many forked worker processes while the
    model runs here. Preprocessing is ~97% of the per-study cost on Kaggle's
    CPU (2.67s vs 0.09s for the model; 3 workers: 2.76 -> 1.17 s/study,
    inference_timing kernel, 2026-09-25). Output order and content are
    identical to n_workers=1 (ordered imap). "fork" explicitly: notebook code
    has no __main__ guard, so "spawn" would re-run the driver in each worker.
    """
    with open(os.path.join(comp_dir, "test.csv"), newline="") as f:
        study_ids = [row[ID_COLUMN] for row in csv.DictReader(f)]
    series_by_study = read_series_csv(os.path.join(comp_dir, "test_series.csv"))

    model.eval()
    rows, seconds, all_errors, n_fallback = [], [], {}, 0
    tasks = [(os.path.join(comp_dir, "test_series", sid), series_by_study.get(sid, [])) for sid in study_ids]
    pool = get_context("fork").Pool(n_workers) if n_workers > 1 else None
    results = pool.imap(_prep_task, tasks) if pool else map(_prep_task, tasks)
    t_start = t0 = time.time()
    for i, (study_id, (image, present, errors)) in enumerate(zip(study_ids, results), 1):
        probs = None
        if errors:
            all_errors[study_id] = errors
        try:
            if image is not None and any(present):
                probs = predict_study(model, image, present, device)
        except Exception as e:  # noqa: BLE001
            all_errors.setdefault(study_id, []).append(f"model: {e!r}")
        if probs is None or not np.all(np.isfinite(probs)):
            probs = np.full(len(columns), FALLBACK_PROB, dtype=np.float32)
            n_fallback += 1
        rows.append([study_id] + [f"{float(np.clip(v, 0.0, 1.0)):.6f}" for v in probs])
        seconds.append(time.time() - t0)  # wall time per study, incl. waiting on workers
        t0 = time.time()
        if i % 10 == 0 or i == len(study_ids):
            log(f"{i}/{len(study_ids)} studies, {time.time()-t_start:.0f}s elapsed, "
                f"{np.mean(seconds):.1f}s/study, {n_fallback} fallback rows")
    if pool:
        pool.close()
        pool.join()

    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([ID_COLUMN] + list(columns))
        w.writerows(rows)
    return {"n_studies": len(study_ids), "n_fallback": n_fallback, "errors": all_errors,
            "seconds_per_study": seconds, "study_ids": study_ids}
