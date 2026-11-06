"""Unit tests for src/training/metrics.py, against hand-computed AUC values.

Hand computations (verify independently, don't trust the code that produced
them):

- y_true=[0,0,1,1], y_pred=[0.1,0.2,0.8,0.9] -> AUC=1.0 (positives rank above
  all negatives).
- y_true=[0,0,1,1], y_pred=[0.9,0.8,0.2,0.1] -> AUC=0.0 (positives rank below
  all negatives).
- y_true=[0,0,1,1], y_pred=[0.1,0.4,0.35,0.8] -> AUC=0.75. Ranks ascending:
  0.1->1, 0.35->2, 0.4->3, 0.8->4. Positive ranks (0.35, 0.8) sum to 6.
  U = 6 - 2*3/2 = 3. AUC = U / (n_pos*n_neg) = 3/4 = 0.75.
- All predictions tied -> AUC=0.5 regardless of labels (every rank is the
  average rank 2.5; U = 5 - 3 = 2; AUC = 2/4 = 0.5).
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from training.metrics import (
    LABEL_COLUMNS,
    bootstrap_macro_auc_ci,
    kaggle_macro_auc,
    macro_auc,
    masked_macro_auc,
    per_column_auc,
)

# Three columns sharing y_true=[0,0,1,1], with known per-column AUCs.
MIXED_COLUMNS = ["perfect", "anti_perfect", "point_75"]
MIXED_Y_TRUE = np.array(
    [[0, 0, 0], [0, 0, 0], [1, 1, 1], [1, 1, 1]], dtype=float
)
MIXED_Y_PRED = np.array(
    [
        [0.1, 0.9, 0.1],
        [0.2, 0.8, 0.4],
        [0.8, 0.2, 0.35],
        [0.9, 0.1, 0.8],
    ],
    dtype=float,
)
MIXED_EXPECTED_AUC = {"perfect": 1.0, "anti_perfect": 0.0, "point_75": 0.75}


def test_per_column_auc_matches_hand_computed_values():
    result = per_column_auc(MIXED_Y_TRUE, MIXED_Y_PRED, MIXED_COLUMNS)
    for col, expected in MIXED_EXPECTED_AUC.items():
        assert result[col] == pytest.approx(expected)


def test_macro_auc_is_plain_mean_when_no_degenerate_columns():
    macro, per_col, degenerate = macro_auc(MIXED_Y_TRUE, MIXED_Y_PRED, MIXED_COLUMNS)
    assert degenerate == []
    expected = (1.0 + 0.0 + 0.75) / 3
    assert abs(macro - expected) < 1e-9


def test_macro_auc_skips_degenerate_column():
    columns = MIXED_COLUMNS + ["all_negative"]
    y_true = np.hstack([MIXED_Y_TRUE, np.zeros((4, 1))])
    y_pred = np.hstack([MIXED_Y_PRED, np.array([[0.3], [0.6], [0.2], [0.9]])])

    macro, per_col, degenerate = macro_auc(y_true, y_pred, columns)

    assert degenerate == ["all_negative"]
    assert np.isnan(per_col["all_negative"])
    expected = (1.0 + 0.0 + 0.75) / 3  # degenerate column excluded, not zero
    assert abs(macro - expected) < 1e-9


def test_macro_auc_all_degenerate_returns_nan():
    y_true = np.array([[0, 1], [0, 1], [0, 1]], dtype=float)  # constant per column
    y_pred = np.array([[0.1, 0.9], [0.5, 0.5], [0.9, 0.1]], dtype=float)

    macro, per_col, degenerate = macro_auc(y_true, y_pred, ["a", "b"])

    assert np.isnan(macro)
    assert set(degenerate) == {"a", "b"}


def test_masked_macro_auc_matches_macro_auc_when_nothing_masked():
    mask = np.ones_like(MIXED_Y_TRUE, dtype=bool)
    macro, per_col, degenerate = masked_macro_auc(MIXED_Y_TRUE, MIXED_Y_PRED, mask, MIXED_COLUMNS)
    expected_macro, expected_per_col, expected_degenerate = macro_auc(MIXED_Y_TRUE, MIXED_Y_PRED, MIXED_COLUMNS)
    assert macro == pytest.approx(expected_macro)
    assert per_col == pytest.approx(expected_per_col)
    assert degenerate == expected_degenerate


def test_masked_macro_auc_uses_a_different_row_set_per_column():
    # column "perfect" masks out row 0 (still 1 neg + 2 pos left -> still perfectly
    # separable, AUC unchanged at 1.0); column "anti_perfect" masks out row 2 (a
    # positive), leaving y_true=[0,0,1], y_pred=[0.9,0.8,0.1] -> still AUC 0.0.
    # "point_75" is left fully unmasked -> AUC unchanged at 0.75.
    mask = np.array([
        [False, True, True],
        [True, True, True],
        [True, False, True],
        [True, True, True],
    ])
    macro, per_col, degenerate = masked_macro_auc(MIXED_Y_TRUE, MIXED_Y_PRED, mask, MIXED_COLUMNS)
    assert degenerate == []
    assert per_col["perfect"] == pytest.approx(1.0)
    assert per_col["anti_perfect"] == pytest.approx(0.0)
    assert per_col["point_75"] == pytest.approx(0.75)


def test_masked_macro_auc_column_with_all_rows_masked_out_is_degenerate():
    mask = np.ones_like(MIXED_Y_TRUE, dtype=bool)
    mask[:, 0] = False  # "perfect" column has zero usable rows
    macro, per_col, degenerate = masked_macro_auc(MIXED_Y_TRUE, MIXED_Y_PRED, mask, MIXED_COLUMNS)
    assert "perfect" in degenerate
    assert np.isnan(per_col["perfect"])
    expected = (0.0 + 0.75) / 2
    assert macro == pytest.approx(expected)


def test_masked_macro_auc_column_that_becomes_single_class_after_masking():
    # mask out both negatives from "perfect" (rows 0,1) -> only positives remain -> degenerate
    mask = np.ones_like(MIXED_Y_TRUE, dtype=bool)
    mask[0, 0] = False
    mask[1, 0] = False
    macro, per_col, degenerate = masked_macro_auc(MIXED_Y_TRUE, MIXED_Y_PRED, mask, MIXED_COLUMNS)
    assert "perfect" in degenerate


def test_masked_macro_auc_rejects_shape_mismatch():
    mask = np.ones((3, 3), dtype=bool)  # wrong n_samples
    with pytest.raises(ValueError):
        masked_macro_auc(MIXED_Y_TRUE, MIXED_Y_PRED, mask, MIXED_COLUMNS)


def test_kaggle_macro_auc_fills_degenerate_with_fill_value():
    columns = MIXED_COLUMNS + ["all_negative"]
    y_true = np.hstack([MIXED_Y_TRUE, np.zeros((4, 1))])
    y_pred = np.hstack([MIXED_Y_PRED, np.array([[0.3], [0.6], [0.2], [0.9]])])

    macro, per_col, n_imputed = kaggle_macro_auc(y_true, y_pred, columns, fill_value=0.5)

    assert n_imputed == 1
    assert per_col["all_negative"] == 0.5
    expected = (1.0 + 0.0 + 0.75 + 0.5) / 4
    assert abs(macro - expected) < 1e-9


def test_full_12_column_formula_matches_confirmed_metric():
    # Every one of the 12 real label columns gets the same 0.75 pattern;
    # macro must equal exactly 0.75, matching (1/12) * sum(AUC_i) = (1/12)*12*0.75.
    assert len(LABEL_COLUMNS) == 12
    base_true = np.array([0, 0, 1, 1], dtype=float)
    base_pred = np.array([0.1, 0.4, 0.35, 0.8], dtype=float)
    y_true = np.tile(base_true.reshape(-1, 1), (1, 12))
    y_pred = np.tile(base_pred.reshape(-1, 1), (1, 12))

    macro, per_col, degenerate = macro_auc(y_true, y_pred, LABEL_COLUMNS)

    assert degenerate == []
    assert abs(macro - 0.75) < 1e-9
    for col in LABEL_COLUMNS:
        assert abs(per_col[col] - 0.75) < 1e-9


def test_all_tied_predictions_give_auc_half():
    y_true = np.array([[0], [1], [0], [1]], dtype=float)
    y_pred = np.array([[0.5], [0.5], [0.5], [0.5]], dtype=float)

    result = per_column_auc(y_true, y_pred, ["tied"])

    assert abs(result["tied"] - 0.5) < 1e-9


def test_shape_mismatch_raises():
    y_true = np.zeros((4, 3))
    y_pred = np.zeros((4, 2))
    with pytest.raises(ValueError):
        macro_auc(y_true, y_pred, ["a", "b", "c"])


def test_wrong_column_count_raises():
    y_true = np.zeros((4, 3))
    y_pred = np.zeros((4, 3))
    with pytest.raises(ValueError):
        macro_auc(y_true, y_pred, ["a", "b"])


def test_bootstrap_ci_point_estimate_matches_macro_auc_on_original_data():
    columns = MIXED_COLUMNS + ["all_negative"]
    y_true = np.hstack([MIXED_Y_TRUE, np.zeros((4, 1))])
    y_pred = np.hstack([MIXED_Y_PRED, np.array([[0.3], [0.6], [0.2], [0.9]])])

    result = bootstrap_macro_auc_ci(y_true, y_pred, columns, n_resamples=200, seed=0)
    direct_macro, _, _ = macro_auc(y_true, y_pred, columns)

    assert abs(result["point_estimate"] - direct_macro) < 1e-9
    assert result["n_resamples_requested"] == 200
    assert result["n_resamples_used"] + result["n_resamples_all_nan"] == 200
    assert 0.0 <= result["ci_low"] <= result["ci_high"] <= 1.0


def test_bootstrap_ci_reproducible_with_fixed_seed():
    columns = MIXED_COLUMNS
    result_a = bootstrap_macro_auc_ci(MIXED_Y_TRUE, MIXED_Y_PRED, columns, n_resamples=100, seed=42)
    result_b = bootstrap_macro_auc_ci(MIXED_Y_TRUE, MIXED_Y_PRED, columns, n_resamples=100, seed=42)

    assert result_a == result_b
