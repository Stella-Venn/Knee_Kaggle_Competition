"""Tests for src/training/engine.py, including the full synthetic smoke-test
pipeline: [redacted: private project notes]'s "build a fast smoke-test pipeline (tiny data subset, 1
epoch, minutes not hours) before trusting any full training run" working
agreement, as an actual automated test rather than a script someone has to
remember to run. Uses synthetic cached volumes (bypasses DICOM/manifest
entirely, same as test_dataset.py's synthetic cases) so it needs no real data
and runs in well under a minute.
"""

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from data.dataset import KneeStudyDataset
from data.label_mining.final_labels import MASKED
from data.sampler import compute_pos_weights, fold_split
from models.knee_model import KneeMultiViewModel
from training.engine import git_commit_hash, predict, set_seed, train_one_epoch
from training.loss import MaskedBCEWithLogitsLoss
from training.metrics import masked_macro_auc
from utils.constants import LABEL_COLUMNS


def test_set_seed_gives_reproducible_torch_init():
    set_seed(42)
    a = torch.randn(5)
    set_seed(42)
    b = torch.randn(5)
    torch.testing.assert_close(a, b)


def test_git_commit_hash_never_raises_and_returns_a_string():
    result = git_commit_hash()
    assert isinstance(result, str) and len(result) > 0


# --- synthetic smoke pipeline ---------------------------------------------

TARGET_SHAPE = (4, 8, 8)
N_STUDIES = 40


def _build_synthetic_cache(tmp_path, n=N_STUDIES, seed=0):
    """n studies, each with all 3 canonical planes cached. Label column 0 has
    a real signal: the cached volume's mean is shifted up for positives, down
    for negatives, so a model that actually learns should separate them.
    Column 1 is deliberately all-negative (checks the NaN-pos_weight and
    degenerate-AUC path together). Column 2 has some MASKED cells. The rest
    are random with no signal.
    """
    rng = np.random.default_rng(seed)
    n_cols = len(LABEL_COLUMNS)
    labels = np.full((n, n_cols), MASKED, dtype=np.int8)
    study_ids = [f"study{i}" for i in range(n)]
    series_by_study = {}

    for i, sid in enumerate(study_ids):
        y0 = int(rng.random() < 0.5)
        labels[i, 0] = y0
        labels[i, 1] = 0  # all-negative column
        labels[i, 2] = int(rng.random() < 0.5) if rng.random() < 0.7 else MASKED
        for j in range(3, n_cols):
            labels[i, j] = int(rng.random() < 0.5)

        offset = 3.0 if y0 == 1 else -3.0
        study_dir = tmp_path / sid
        study_dir.mkdir()
        series_rows = []
        for plane in ("Sagittal", "Coronal", "Axial"):
            series_id = f"{sid}_{plane}"
            volume = rng.normal(loc=offset, scale=0.5, size=TARGET_SHAPE).astype(np.float32)
            np.save(study_dir / f"{series_id}.npy", volume)
            series_rows.append({
                "SeriesInstanceUID": series_id, "n_slices": TARGET_SHAPE[0],
                "csv_Anatomical_Plane": plane,
            })
        series_by_study[sid] = series_rows

    return study_ids, labels, series_by_study


def test_smoke_pipeline_trains_and_learns_the_synthetic_signal(tmp_path):
    set_seed(0)
    study_ids, labels, series_by_study = _build_synthetic_cache(tmp_path)

    # simple 50/50 split; not using splits.make_cv_folds here since that's
    # about site-stratification/patient-grouping, orthogonal to what this
    # test checks (does the train/predict loop actually learn).
    fold = np.array([i % 2 for i in range(len(study_ids))])
    train_idx, val_idx = fold_split(study_ids, fold, val_fold=0)

    full_ds = KneeStudyDataset(study_ids, labels, series_by_study, cache_dir=tmp_path,
                                target_shape=TARGET_SHAPE)
    train_loader = torch.utils.data.DataLoader(
        torch.utils.data.Subset(full_ds, train_idx), batch_size=8, shuffle=True,
    )
    val_loader = torch.utils.data.DataLoader(
        torch.utils.data.Subset(full_ds, val_idx), batch_size=8, shuffle=False,
    )

    model = KneeMultiViewModel(
        n_classes=len(LABEL_COLUMNS), in_channels=1, stem_channels=4, channels=(8,),
    )
    pos_weight = compute_pos_weights(labels[train_idx])
    loss_fn = MaskedBCEWithLogitsLoss(pos_weight=pos_weight, nan_pos_weight_fallback=1.0)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)

    losses = [train_one_epoch(model, train_loader, optimizer, loss_fn, device="cpu") for _ in range(15)]

    assert all(np.isfinite(losses))
    # Aggregate loss is a mean over all 12 columns, and 10 of them (3..11) are
    # pure random noise by construction -- irreducible, so it stays near
    # ln(2) regardless of training and isn't the right thing to assert on.
    # A modest overall improvement is still expected; the real proof is the
    # per-column AUC below.
    assert losses[-1] < losses[0]

    study_ids_out, y_true, y_pred, mask = predict(model, val_loader, device="cpu")
    assert set(study_ids_out) == {study_ids[i] for i in val_idx}
    assert y_pred.shape == (len(val_idx), len(LABEL_COLUMNS))
    assert np.isfinite(y_pred).all()

    macro, per_col, degenerate = masked_macro_auc(y_true, y_pred, mask, LABEL_COLUMNS)
    j1 = LABEL_COLUMNS[1]
    assert np.isnan(per_col[j1])  # all-negative column stayed degenerate through training
    j0 = LABEL_COLUMNS[0]
    assert per_col[j0] > 0.9  # the model learned the strong mean-offset signal (measured 1.0)


def test_predict_matches_direct_forward_pass(tmp_path):
    set_seed(0)
    study_ids, labels, series_by_study = _build_synthetic_cache(tmp_path, n=6)
    ds = KneeStudyDataset(study_ids, labels, series_by_study, cache_dir=tmp_path, target_shape=TARGET_SHAPE)
    loader = torch.utils.data.DataLoader(ds, batch_size=6, shuffle=False)
    model = KneeMultiViewModel(n_classes=len(LABEL_COLUMNS), in_channels=1, stem_channels=2, channels=(4,))
    model.eval()

    study_ids_out, y_true, y_pred, mask = predict(model, loader, device="cpu")

    batch = next(iter(loader))
    with torch.no_grad():
        expected = torch.sigmoid(model(batch["image"], batch["plane_present"])).numpy()
    np.testing.assert_allclose(y_pred, expected, atol=1e-6)
    assert study_ids_out == batch["study_id"]
