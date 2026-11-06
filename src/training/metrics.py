"""Macro-AUC metric: (1/12) * sum(AUC_i) over the 12 label columns.

A gold-labeled evaluation set can have zero positives (or zero negatives) in a
column -- with only 58 gold-labeled studies across 12 classes this is expected,
not rare. AUC is undefined for such a column. `macro_auc` treats it as NaN
internally so degenerate columns are visible and countable rather than
silently distorting the average; `kaggle_macro_auc` fills degenerate columns
with `fill_value` instead, as our best guess at how the hidden Kaggle scorer
behaves (unconfirmed) -- use that variant only when trying to emulate that
scorer, and check `n_imputed` to see whether the guess is even being exercised.
"""

import numpy as np
from sklearn.metrics import roc_auc_score

from utils.constants import LABEL_COLUMNS


def _validate_shapes(y_true, y_pred, columns):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if y_true.shape != y_pred.shape:
        raise ValueError(f"y_true shape {y_true.shape} != y_pred shape {y_pred.shape}")
    if y_true.ndim != 2 or y_true.shape[1] != len(columns):
        raise ValueError(
            f"expected shape (n_samples, {len(columns)}), got {y_true.shape}"
        )
    return y_true, y_pred


def per_column_auc(y_true, y_pred, columns=LABEL_COLUMNS):
    """AUC per column; NaN for a column with a single class in y_true."""
    y_true, y_pred = _validate_shapes(y_true, y_pred, columns)
    result = {}
    for j, col in enumerate(columns):
        col_true = y_true[:, j]
        if np.unique(col_true).size < 2:
            result[col] = float("nan")
        else:
            result[col] = float(roc_auc_score(col_true, y_pred[:, j]))
    return result


def macro_auc(y_true, y_pred, columns=LABEL_COLUMNS):
    """NaN-aware macro AUC: mean over columns with a defined AUC.

    Returns (macro_score, per_column_dict, degenerate_columns). macro_score is
    NaN only if every column is degenerate.
    """
    per_col = per_column_auc(y_true, y_pred, columns)
    values = np.array(list(per_col.values()), dtype=float)
    degenerate = [c for c, v in per_col.items() if np.isnan(v)]
    macro = float("nan") if np.all(np.isnan(values)) else float(np.nanmean(values))
    return macro, per_col, degenerate


def masked_macro_auc(y_true, y_pred, mask, columns=LABEL_COLUMNS):
    """macro_auc's counterpart for a label table with per-cell missingness
    that varies BY COLUMN (module 6/8's training label table: e.g. a study
    mined for Medial OA is very often not mined for ACL) -- a single
    row-level filter can't handle this, since the set of usable rows differs
    per column.

    `mask`: (n_samples, len(columns)) boolean, True where y_true[i, j] is a
    real 0/1 label. y_true's value at a masked-out cell is never read.
    Deliberately takes an explicit mask rather than a sentinel value (e.g.
    label_mining.final_labels.MASKED) so this module stays independent of
    that convention -- metrics.py has no dependency on label_mining today,
    and this keeps it that way.

    Same NaN-for-degenerate-column convention as `macro_auc`: a column whose
    masked-in rows are single-class (including "zero masked-in rows") gets
    NaN, not a fabricated score. Returns (macro_score, per_column_dict,
    degenerate_columns).
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = np.asarray(mask, dtype=bool)
    if not (y_true.shape == y_pred.shape == mask.shape):
        raise ValueError(
            f"y_true, y_pred, mask must share one shape; got {y_true.shape}, "
            f"{y_pred.shape}, {mask.shape}"
        )
    if y_true.ndim != 2 or y_true.shape[1] != len(columns):
        raise ValueError(f"expected shape (n_samples, {len(columns)}), got {y_true.shape}")

    per_col = {}
    for j, col in enumerate(columns):
        rows = mask[:, j]
        col_true = y_true[rows, j]
        col_pred = y_pred[rows, j]
        if np.unique(col_true).size < 2:
            per_col[col] = float("nan")
        else:
            per_col[col] = float(roc_auc_score(col_true, col_pred))
    values = np.array(list(per_col.values()), dtype=float)
    degenerate = [c for c, v in per_col.items() if np.isnan(v)]
    macro = float("nan") if np.all(np.isnan(values)) else float(np.nanmean(values))
    return macro, per_col, degenerate


def kaggle_macro_auc(y_true, y_pred, columns=LABEL_COLUMNS, fill_value=0.5):
    """[redacted: detail about the data challenge this code was written for]
    instead of excluded, matching (1/12) * sum(AUC_i) exactly once filled.

    Returns (macro_score, per_column_dict, n_imputed).
    """
    per_col = per_column_auc(y_true, y_pred, columns)
    n_imputed = sum(1 for v in per_col.values() if np.isnan(v))
    filled = {c: (fill_value if np.isnan(v) else v) for c, v in per_col.items()}
    macro = float(np.mean(list(filled.values())))
    return macro, filled, n_imputed


def bootstrap_macro_auc_ci(
    y_true, y_pred, columns=LABEL_COLUMNS, n_resamples=2000, ci=0.95, seed=0
):
    """Row-wise bootstrap CI for macro_auc (NaN-aware at both the per-resample
    column level and across resamples: a resample where every column is
    degenerate contributes no score rather than an arbitrary neutral value).

    Returns a dict: point_estimate (macro_auc on the original data), ci_low,
    ci_high (percentile bounds over resamples with at least one defined
    column), n_resamples_used, n_resamples_all_nan.
    """
    y_true, y_pred = _validate_shapes(y_true, y_pred, columns)
    n = y_true.shape[0]
    rng = np.random.default_rng(seed)
    point_estimate, _, _ = macro_auc(y_true, y_pred, columns)

    scores = []
    n_all_nan = 0
    for _ in range(n_resamples):
        idx = rng.integers(0, n, size=n)
        score, _, _ = macro_auc(y_true[idx], y_pred[idx], columns)
        if np.isnan(score):
            n_all_nan += 1
        else:
            scores.append(score)

    scores = np.array(scores)
    alpha = (1 - ci) / 2
    if len(scores) == 0:
        ci_low = ci_high = float("nan")
    else:
        ci_low = float(np.quantile(scores, alpha))
        ci_high = float(np.quantile(scores, 1 - alpha))

    return {
        "point_estimate": point_estimate,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "n_resamples_requested": n_resamples,
        "n_resamples_used": len(scores),
        "n_resamples_all_nan": n_all_nan,
    }
