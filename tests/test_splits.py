"""Tests for src/data/splits.py."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from data.splits import (
    _GOLD_EVAL_WARN_THRESHOLD,
    evaluate_on_gold,
    make_cv_folds,
    normalize_manufacturer,
    site_proxy_key,
)
from utils.constants import LABEL_COLUMNS


def test_normalize_manufacturer_aliases_and_unknown():
    assert normalize_manufacturer("SIEMENS") == "Siemens"
    assert normalize_manufacturer("Siemens Healthineers") == "Siemens"
    assert normalize_manufacturer("Philips Medical Systems") == "Philips"
    assert normalize_manufacturer("GEHC") == "GE"
    assert normalize_manufacturer("") == "Unknown"
    assert normalize_manufacturer(None) == "Unknown"
    assert normalize_manufacturer("Acme Imaging") == "Acme Imaging"  # unmapped: passed through


def test_site_proxy_key_buckets_field_strength():
    assert site_proxy_key("SIEMENS", 1.5) == "Siemens|1.5T"
    assert site_proxy_key("SIEMENS", 1.51) == "Siemens|1.5T"  # within tolerance
    assert site_proxy_key("SIEMENS", 2.2) == "Siemens|Other"
    assert site_proxy_key("SIEMENS", None) == "Siemens|Unknown"


def test_make_cv_folds_every_study_assigned_and_balanced():
    rng = np.random.default_rng(0)
    n = 500
    study_ids = [f"s{i}" for i in range(n)]
    patient_ids = study_ids  # 1:1, per the confirmed manifest finding
    proxies = rng.choice(["Siemens|1.5T", "GE|3.0T", "Philips|1.5T"], size=n)
    fold = make_cv_folds(study_ids, patient_ids, proxies, n_folds=5, seed=0)
    assert fold.shape == (n,)
    assert set(fold.tolist()) == {0, 1, 2, 3, 4}
    counts = np.bincount(fold)
    assert counts.min() >= n // 5 - 15  # roughly balanced, not exact due to grouping+stratification


def test_make_cv_folds_never_splits_a_patient_across_folds():
    # 3 studies per patient, 30 patients -> no patient's studies may land in >1 fold
    study_ids, patient_ids, proxies = [], [], []
    for p in range(30):
        for k in range(3):
            study_ids.append(f"study{p}_{k}")
            patient_ids.append(f"patient{p}")
            proxies.append("Siemens|1.5T")
    fold = make_cv_folds(study_ids, patient_ids, proxies, n_folds=5, seed=1)
    by_patient = {}
    for pid, f in zip(patient_ids, fold):
        by_patient.setdefault(pid, set()).add(int(f))
    assert all(len(folds) == 1 for folds in by_patient.values())


def test_make_cv_folds_rare_site_bucket_does_not_crash():
    # one rare bucket (n=2) should be pooled into "Rare" rather than breaking StratifiedGroupKFold
    n = 60
    study_ids = [f"s{i}" for i in range(n)]
    patient_ids = study_ids
    proxies = ["Siemens|1.5T"] * 58 + ["Hitachi|1.0T"] * 2
    fold = make_cv_folds(study_ids, patient_ids, proxies, n_folds=5, seed=0, min_group_size=10)
    assert set(fold.tolist()) == {0, 1, 2, 3, 4}


def test_make_cv_folds_rejects_mismatched_lengths():
    with pytest.raises(ValueError):
        make_cv_folds(["a", "b"], ["a"], ["x", "y"])


def test_make_cv_folds_rejects_too_few_studies():
    with pytest.raises(ValueError):
        make_cv_folds(["a", "b"], ["a", "b"], ["x", "y"], n_folds=5)


def test_evaluate_on_gold_matches_bootstrap_ci_and_counts_calls(capsys):
    import data.splits as splits_module

    splits_module._gold_eval_call_count = 0
    n, k = 58, len(LABEL_COLUMNS)
    rng = np.random.default_rng(0)
    y_true = (rng.random((n, k)) > 0.5).astype(int)
    y_pred = rng.random((n, k))
    result = evaluate_on_gold(y_true, y_pred, n_resamples=200, seed=0, label="unit-test-model")
    assert "point_estimate" in result and "ci_low" in result and "ci_high" in result
    err = capsys.readouterr().err
    assert "evaluate_on_gold call #1" in err and "unit-test-model" in err


def test_evaluate_on_gold_warns_past_threshold(capsys):
    import data.splits as splits_module

    splits_module._gold_eval_call_count = 0
    n, k = 58, len(LABEL_COLUMNS)
    rng = np.random.default_rng(0)
    y_true = (rng.random((n, k)) > 0.5).astype(int)
    y_pred = rng.random((n, k))
    for _ in range(_GOLD_EVAL_WARN_THRESHOLD):
        evaluate_on_gold(y_true, y_pred, n_resamples=50)
    err = capsys.readouterr().err
    assert "WARNING" in err
