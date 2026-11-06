"""Validation split module (gameplan module 3, second half).

Two things live here, and they are deliberately not the same code path:

1. `make_cv_folds` -- patient-grouped, scanner-site-stratified K folds over the
   *mined-label pool* (every training study except the gold 58), for frequent,
   cheap, day-to-day model iteration.
2. `evaluate_on_gold` -- scores predictions against the precious gold 58 with a
   bootstrap CI, structurally separate from the CV loop above (a different
   function, in the same module but never called from `make_cv_folds` or by
   any loop that iterates candidate models automatically) so it's hard to
   reach for by accident. It also counts and logs its own calls -- see its
   docstring -- because "structurally separate" alone doesn't stop someone
   from calling it in a hyperparameter sweep.

Why patient-grouping is still done even though 2026-09-21/22's full-dataset
DICOM manifest found `PatientID` maps 1:1 to `StudyInstanceUID` (see
[redacted: private project notes], "DICOM header fields"): grouping by `PatientID` costs nothing, is
correct if that 1:1 mapping ever turns out wrong for a subset we haven't
[redacted: detail about the data challenge this code was written for]
doesn't change fold composition today.

Why stratify by scanner/site instead of by label: with 12 correlated binary
targets, proper multi-label stratification needs iterative stratification
(not in scikit-learn, a new dependency) for a payoff that's unclear given how
imbalanced and correlated the labels already are. Domain shift across
scanners is the specific risk the engineering priorities flagged ([redacted: private project notes]),
and sklearn's `StratifiedGroupKFold` already does grouped+single-label
stratification, so that's what folds are built on for a first pass.
"""

import sys
import warnings

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold

from training.metrics import bootstrap_macro_auc_ci
from utils.constants import LABEL_COLUMNS

# Collapses inconsistent DICOM Manufacturer spellings seen in the full manifest
# (data/processed/manifest/study_manifest.csv, 2026-09-21/22) to one vendor name.
_MANUFACTURER_ALIASES = {
    "siemens": "Siemens", "siemens healthineers": "Siemens",
    "philips": "Philips", "philips healthcare": "Philips", "philips medical systems": "Philips",
    "ge medical systems": "GE", "gehc": "GE",
    "toshiba": "Canon/Toshiba", "canon_mec": "Canon/Toshiba",
    "fujifilm healthcare corporation": "Fujifilm",
    "hitachi medical corporation": "Hitachi",
}
# Nominal field strengths seen in the manifest; anything else buckets to "Other".
_FIELD_STRENGTHS = (1.0, 1.5, 3.0)
_FIELD_STRENGTH_TOLERANCE = 0.1


def normalize_manufacturer(name) -> str:
    """Canonical vendor name from a raw DICOM Manufacturer string. Unknown or
    missing input returns "Unknown", not a guess.
    """
    if not name:
        return "Unknown"
    return _MANUFACTURER_ALIASES.get(str(name).strip().lower(), str(name).strip())


def _field_strength_bucket(value) -> str:
    if value in (None, ""):
        return "Unknown"
    try:
        value = float(value)
    except (TypeError, ValueError):
        return "Unknown"
    for nominal in _FIELD_STRENGTHS:
        if abs(value - nominal) <= _FIELD_STRENGTH_TOLERANCE:
            return f"{nominal:g}T"
    return "Other"


def site_proxy_key(manufacturer, field_strength) -> str:
    """One categorical "site" stand-in from (Manufacturer, MagneticFieldStrength),
    [redacted: detail about the data challenge this code was written for]
    balance fold composition, not as a ground-truth site label.
    """
    return f"{normalize_manufacturer(manufacturer)}|{_field_strength_bucket(field_strength)}"


def make_cv_folds(study_ids, patient_ids, site_proxies, n_folds=5, seed=0, min_group_size=10):
    """Patient-grouped, site-stratified fold assignment for the mined-label pool.

    study_ids, patient_ids, site_proxies: equal-length sequences, one entry per
    study (patient_ids should already reflect the 1:1 PatientID<->StudyInstanceUID
    mapping confirmed dataset-wide; a study's own StudyInstanceUID is a safe
    substitute if PatientID is ever missing for some studies).

    Rare site-proxy buckets (fewer than min_group_size studies) are pooled into
    a single "Rare" stratum first -- StratifiedGroupKFold needs every stratum to
    have at least n_folds members, and 51 (Manufacturer, Model) pairs over 4407
    studies (see [redacted: private project notes]) means several buckets are that small.

    Returns an (n_studies,) int array of fold indices in [0, n_folds), aligned
    to `study_ids`.
    """
    study_ids = list(study_ids)
    n = len(study_ids)
    if not (n == len(patient_ids) == len(site_proxies)):
        raise ValueError("study_ids, patient_ids, site_proxies must be the same length")
    if n < n_folds:
        raise ValueError(f"{n} studies is fewer than n_folds={n_folds}")

    proxies = np.asarray(site_proxies, dtype=object)
    counts = {}
    for p in proxies:
        counts[p] = counts.get(p, 0) + 1
    strata = np.array([p if counts[p] >= max(min_group_size, n_folds) else "Rare" for p in proxies])

    # Pooling rare buckets into "Rare" can still leave that pooled bucket itself
    # under n_folds (e.g. two single-study rare buckets from different vendors).
    # sklearn only warns in that case and falls back to an unstratified split for
    # the affected rows -- expected here, not a bug, so it's suppressed rather
    # than left to alarm whoever runs this for real.
    splitter = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    fold = np.full(n, -1, dtype=int)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="The least populated class.*", category=UserWarning)
        for f, (_, val_idx) in enumerate(splitter.split(np.zeros(n), strata, groups=patient_ids)):
            fold[val_idx] = f
    assert (fold >= 0).all(), "every study should receive a fold"
    return fold


_gold_eval_call_count = 0
_GOLD_EVAL_WARN_THRESHOLD = 10  # matches the "evaluate sparingly" agreement in [redacted: private project notes]


def evaluate_on_gold(y_true, y_pred, columns=LABEL_COLUMNS, n_resamples=2000, seed=0, label=""):
    """Score predictions against the gold-58 with a bootstrap CI (see
    training.metrics.bootstrap_macro_auc_ci). This is the ONLY sanctioned way to
    check a model against the gold set -- per [redacted: private project notes]'s "Validation strategy",
    the gold 58 is precious and should be spent sparingly, not used as routine
    CV. Every call is counted (module-level, resets on process restart) and
    logged to stderr; past `_GOLD_EVAL_WARN_THRESHOLD` calls in one process, a
    warning is printed so a sweep that's accidentally hammering the gold set is
    visible, not silent. This does not block the call -- the discipline is a
    convention this function makes visible, not a hard limit.

    label: optional short string identifying the candidate/run being scored,
    included in the log line so repeated calls are distinguishable.
    """
    global _gold_eval_call_count
    _gold_eval_call_count += 1
    result = bootstrap_macro_auc_ci(y_true, y_pred, columns=columns, n_resamples=n_resamples, seed=seed)
    print(
        f"[evaluate_on_gold call #{_gold_eval_call_count}] {label or '(unlabeled)'}: "
        f"macro_auc={result['point_estimate']:.4f} "
        f"CI=[{result['ci_low']:.4f}, {result['ci_high']:.4f}] "
        f"(n_resamples_used={result['n_resamples_used']}, n_all_nan={result['n_resamples_all_nan']})",
        file=sys.stderr,
    )
    if _gold_eval_call_count == _GOLD_EVAL_WARN_THRESHOLD:
        print(
            f"[evaluate_on_gold] WARNING: {_gold_eval_call_count} calls this process. "
            "The gold 58 is meant to be spent sparingly [redacted: private project notes] "
            "-- if this is a sweep over many candidates, switch it to CV folds "
            "(make_cv_folds) and save this for a final handful of models.",
            file=sys.stderr,
        )
    return result
