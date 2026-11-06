"""Tests for src/data/sampler.py."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from data.label_mining.final_labels import MASKED
from data.sampler import compute_pos_weights, fold_split


# --- fold_split -----------------------------------------------------------

def test_fold_split_partitions_correctly():
    study_ids = [f"s{i}" for i in range(10)]
    fold = np.array([0, 1, 0, 1, 2, 2, 0, 1, 2, 0])
    train_idx, val_idx = fold_split(study_ids, fold, val_fold=1)
    assert set(val_idx.tolist()) == {1, 3, 7}
    assert set(train_idx.tolist()) == set(range(10)) - {1, 3, 7}


def test_fold_split_rejects_mismatched_lengths():
    with pytest.raises(ValueError):
        fold_split(["a", "b"], [0], val_fold=0)


def test_fold_split_rejects_missing_fold():
    with pytest.raises(ValueError):
        fold_split(["a", "b"], [0, 0], val_fold=5)


# --- compute_pos_weights ----------------------------------------------------

def test_compute_pos_weights_hand_computed():
    columns = ["A", "B"]
    # column A = [1,1,0,0,0,0] -> 2 pos, 4 neg -> weight 4/2 = 2.0
    # column B = [1,0,1,0,0,MASKED] -> (masked row excluded) 2 pos, 3 neg -> weight 3/2 = 1.5
    labels = np.array([
        [1, 1], [1, 0], [0, 1], [0, 0], [0, 0], [0, MASKED],
    ])
    w = compute_pos_weights(labels, columns=columns)
    assert w[0] == pytest.approx(4 / 2)
    assert w[1] == pytest.approx(3 / 2)


def test_compute_pos_weights_masked_cells_excluded():
    columns = ["A"]
    labels = np.array([[1], [MASKED], [MASKED], [0], [0]])
    w = compute_pos_weights(labels, columns=columns)
    assert w[0] == pytest.approx(2 / 1)  # only the 3 non-masked cells count


def test_compute_pos_weights_zero_positives_is_nan():
    columns = ["A"]
    labels = np.array([[0], [0], [MASKED]])
    w = compute_pos_weights(labels, columns=columns)
    assert np.isnan(w[0])


def test_compute_pos_weights_zero_negatives_uses_max_weight():
    columns = ["A"]
    labels = np.array([[1], [1], [1]])
    w = compute_pos_weights(labels, columns=columns, max_weight=7.0)
    assert w[0] == 7.0


def test_compute_pos_weights_caps_extreme_ratio():
    columns = ["A"]
    labels = np.array([[1]] + [[0]] * 100)
    w = compute_pos_weights(labels, columns=columns, max_weight=10.0)
    assert w[0] == 10.0


def test_compute_pos_weights_rejects_wrong_column_count():
    with pytest.raises(ValueError):
        compute_pos_weights(np.zeros((3, 2)), columns=["A", "B", "C"])


def test_compute_pos_weights_uses_weighted_counts():
    labels = np.full((5, 12), MASKED, dtype=np.int8)
    labels[:, 0] = [1, 0, 0, 0, 0]
    cw = np.ones(labels.shape, dtype=np.float32)
    cw[1:, 0] = 0.25  # four negatives at weight 0.25 == one negative
    assert compute_pos_weights(labels)[0] == pytest.approx(4.0)
    assert compute_pos_weights(labels, cell_weights=cw)[0] == pytest.approx(1.0)
