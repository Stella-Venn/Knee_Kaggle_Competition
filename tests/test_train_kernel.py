"""Tests for the training-only helpers that live in the GENERATED Kaggle
training kernel (kaggle_kernels/train_full/train_kernel.py) -- augmentation,
LR schedule, EMA (2026-09-24). Loads the generated file itself, so what's
tested is exactly what ships to Kaggle. Regenerate with
scripts/build_train_kernel.py after editing the generator.
"""

import importlib.util
from pathlib import Path

import numpy as np
import pytest
import torch

KERNEL = Path(__file__).resolve().parents[1] / "kaggle_kernels" / "train_full" / "train_kernel.py"


@pytest.fixture(scope="module")
def k():
    if not KERNEL.exists():
        pytest.skip("generated training notebook not present (it embeds dataset rows, so it "
                    "is never committed; build it with scripts/build_train_kernel.py)")
    spec = importlib.util.spec_from_file_location("train_kernel", KERNEL)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_shift_zero_fill(k):
    v = np.arange(27, dtype=np.float32).reshape(3, 3, 3) + 1
    out = k._shift_zero_fill(v, (1, 0, -1))
    assert out.shape == v.shape
    assert (out[0] == 0).all()                      # shifted in along z: zero-filled
    assert (out[1:, :, :2] == v[:2, :, 1:]).all()   # content moved, not wrapped
    assert (out[:, :, 2] == 0).all()


def test_augment_volume_keeps_shape_dtype_range(k):
    rng = np.random.default_rng(0)
    v = rng.random((32, 64, 64), dtype=np.float32)
    for _ in range(10):
        out = k.augment_volume(v, rng)
        assert out.shape == v.shape and out.dtype == np.float32
        assert out.min() >= 0.0 and out.max() <= 1.0


def test_warmup_cosine_schedule(k):
    total, warm = 100, 10
    assert k.warmup_cosine(0, total, warm) == pytest.approx(0.1)
    assert k.warmup_cosine(warm, total, warm) == pytest.approx(1.0)
    assert k.warmup_cosine(total, total, warm) == pytest.approx(0.02)
    vals = [k.warmup_cosine(s, total, warm) for s in range(warm, total + 1)]
    assert all(a >= b for a, b in zip(vals, vals[1:]))  # monotone decay after warm-up


def test_model_ema_averages_params_and_copies_buffers(k):
    model = torch.nn.Sequential(torch.nn.Linear(2, 1), torch.nn.BatchNorm1d(1))
    ema = k.ModelEMA(model, decay=0.5)
    with torch.no_grad():
        model[0].weight.fill_(1.0)
        model[1].running_mean.fill_(3.0)
    before = ema.module[0].weight.clone()
    ema.update(model)
    assert torch.allclose(ema.module[0].weight, 0.5 * before + 0.5 * 1.0)
    assert torch.allclose(ema.module[1].running_mean, torch.tensor([3.0]))  # copied, not averaged


def test_strict_val_exclusions_match_label_table_sources(k):
    """STUDY_FILLED (cells excluded from the strict val metric) must be exactly
    the SOURCE_UNMENTIONED cells in training_labels.npz, for every embedded
    study -- the generator hand-copies the source code, so this also guards
    against final_labels renumbering it."""
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from data.label_mining.final_labels import SOURCE_UNMENTIONED
    t = np.load(Path(__file__).resolve().parents[1] / "data" / "processed" / "training_labels.npz",
                allow_pickle=True)
    idx = {s: i for i, s in enumerate(t["study_ids"])}
    for sid in k.STUDY_LABELS:
        expected = [int(j) for j in np.where(t["source"][idx[sid]] == SOURCE_UNMENTIONED)[0]]
        assert k.STUDY_FILLED.get(sid, []) == expected, sid


def test_compact_embedding_decodes_to_label_table_and_folds(k):
    """The kernel embeds one "fold|labels|weights|filled" row per study
    (2026-09-25, to fit Kaggle's push size limit); the decoded dicts must
    equal the label table / CV folds exactly."""
    root = Path(__file__).resolve().parents[1]
    t = np.load(root / "data" / "processed" / "training_labels.npz", allow_pickle=True)
    f = np.load(root / "data" / "processed" / "cv_folds.npz", allow_pickle=True)
    idx = {s: i for i, s in enumerate(t["study_ids"])}
    fold = dict(zip(f["study_ids"].tolist(), f["fold"].tolist()))
    assert set(k.STUDY_LABELS) == set(fold)  # full mined pool
    for sid, labels in k.STUDY_LABELS.items():
        i = idx[sid]
        assert labels == t["labels"][i].tolist(), sid
        assert k.STUDY_FOLD[sid] == fold[sid], sid
        w = t["weights"][i]
        if (w != 1.0).any():
            assert np.allclose(k.STUDY_WEIGHTS[sid], w), sid
        else:
            assert sid not in k.STUDY_WEIGHTS, sid
