"""Sampler half of gameplan module 6: fold-based train/val index splits (using
splits.py's make_cv_folds output) and masked-aware per-label class weights for
the training loss.

Not built here, and why: a per-sample weighted *sampler* for class imbalance.
With 12 correlated binary targets at very different prevalences (e.g. MCL 6%
positive vs. Synovitis 89% positive among labeled cells), one sampling weight
per study can't simultaneously fix every column's imbalance -- the standard
fix for multi-label imbalance is a per-label weight in the loss
(BCEWithLogitsLoss's `pos_weight`), computed here by `compute_pos_weights`,
not a resampling scheme. This keeps the imbalance fix at the loss (module 8)
where it can be applied per-column, not smeared across one sampling
distribution.
"""

import numpy as np

from data.label_mining.final_labels import MASKED
from utils.constants import LABEL_COLUMNS


def fold_split(study_ids, fold_assignment, val_fold):
    """(train_idx, val_idx): positions into `study_ids` for a single-fold
    train/val split, matching `make_cv_folds`'s output convention (fold
    indices in [0, n_folds)). Returns index arrays, not study IDs, since
    that's what a torch Subset/DataLoader needs directly.
    """
    fold_assignment = np.asarray(fold_assignment)
    if len(fold_assignment) != len(study_ids):
        raise ValueError(f"{len(study_ids)} study_ids but {len(fold_assignment)} fold assignments")
    if val_fold not in fold_assignment:
        raise ValueError(f"val_fold={val_fold} not present in fold_assignment")
    val_idx = np.where(fold_assignment == val_fold)[0]
    train_idx = np.where(fold_assignment != val_fold)[0]
    return train_idx, val_idx


def compute_pos_weights(labels, columns=LABEL_COLUMNS, max_weight=50.0, cell_weights=None):
    """Per-column `pos_weight` for `torch.nn.BCEWithLogitsLoss` (weight on the
    positive class = n_negative / n_positive among non-MASKED cells), so a
    label like MCL (178 pos / 2817 neg in the current training table, ~6%
    positive) doesn't get drowned out by columns like Baker's that are closer
    to balanced.

    A column with zero known positives gets NaN, not a fabricated weight --
    visible and countable (same convention as training.metrics.macro_auc's
    NaN-for-degenerate-columns), since "how to handle a positive-free column"
    is a training-loop decision (e.g. fall back to weight=1.0, or exclude the
    column from the loss that epoch), not one to silently make here.
    `max_weight` caps an otherwise-extreme ratio (e.g. a fold with very few
    positives) so one column can't dominate a multi-label loss.
    `cell_weights` (same shape as labels, optional): counts become weighted
    sums, matching the per-cell loss weights -- 3994 Synovitis PU negatives
    at weight 0.3 count as ~1198 negatives, not 3994.
    """
    labels = np.asarray(labels)
    if labels.shape[1] != len(columns):
        raise ValueError(f"labels has {labels.shape[1]} columns, expected {len(columns)}")
    weights = np.full(len(columns), np.nan, dtype=np.float64)
    cw = np.ones(labels.shape) if cell_weights is None else np.asarray(cell_weights, dtype=np.float64)
    for j in range(len(columns)):
        col, w = labels[:, j], cw[:, j]
        n_pos = float(w[col == 1].sum())
        n_neg = float(w[col == 0].sum())
        if n_pos > 0:
            weights[j] = min(n_neg / n_pos, max_weight) if n_neg > 0 else max_weight
    return weights
