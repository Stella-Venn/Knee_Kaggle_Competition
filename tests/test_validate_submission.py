"""Tests for src/inference/validate_submission.py."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from inference.validate_submission import (
    ID_COLUMN,
    expected_ids_from_csv,
    load_submission_csv,
    validate_submission,
    validate_submission_file,
)
from utils.constants import LABEL_COLUMNS

EXPECTED_IDS = ["s1", "s2", "s3"]
HEADER = [ID_COLUMN] + LABEL_COLUMNS


def _row(sid, value=0.5):
    return {ID_COLUMN: sid, **{c: str(value) for c in LABEL_COLUMNS}}


def test_valid_submission_passes():
    rows = [_row(sid) for sid in EXPECTED_IDS]
    is_valid, issues = validate_submission(rows, HEADER, EXPECTED_IDS)
    assert is_valid and issues == []


def test_wrong_column_order_flagged():
    bad_header = [ID_COLUMN] + list(reversed(LABEL_COLUMNS))
    rows = [_row(sid) for sid in EXPECTED_IDS]
    is_valid, issues = validate_submission(rows, bad_header, EXPECTED_IDS)
    assert not is_valid
    assert any("column names/order mismatch" in i for i in issues)


def test_row_count_mismatch_flagged():
    rows = [_row(sid) for sid in EXPECTED_IDS[:2]]
    is_valid, issues = validate_submission(rows, HEADER, EXPECTED_IDS)
    assert not is_valid
    assert any("row count mismatch" in i for i in issues)


def test_missing_study_id_flagged():
    rows = [_row(sid) for sid in EXPECTED_IDS[:2]]
    is_valid, issues = validate_submission(rows, HEADER, EXPECTED_IDS)
    assert any("missing 1 expected" in i for i in issues)


def test_extra_study_id_flagged():
    rows = [_row(sid) for sid in EXPECTED_IDS] + [_row("s4")]
    is_valid, issues = validate_submission(rows, HEADER, EXPECTED_IDS)
    assert not is_valid
    assert any("unexpected" in i for i in issues)


def test_duplicate_study_id_flagged():
    rows = [_row("s1"), _row("s1"), _row("s2")]
    is_valid, issues = validate_submission(rows, HEADER, ["s1", "s2"])
    assert not is_valid
    assert any("duplicate" in i for i in issues)


def test_reordered_ids_flagged_but_missing_extra_dup_clean():
    rows = [_row(sid) for sid in reversed(EXPECTED_IDS)]
    is_valid, issues = validate_submission(rows, HEADER, EXPECTED_IDS)
    assert not is_valid
    assert any("order differs" in i for i in issues)
    assert not any("missing" in i or "unexpected" in i or "duplicate" in i for i in issues)


def test_missing_value_flagged():
    rows = [_row(sid) for sid in EXPECTED_IDS]
    rows[0][LABEL_COLUMNS[0]] = ""
    is_valid, issues = validate_submission(rows, HEADER, EXPECTED_IDS)
    assert not is_valid
    assert any("missing value" in i for i in issues)


def test_non_numeric_value_flagged():
    rows = [_row(sid) for sid in EXPECTED_IDS]
    rows[1][LABEL_COLUMNS[2]] = "high"
    is_valid, issues = validate_submission(rows, HEADER, EXPECTED_IDS)
    assert not is_valid
    assert any("non-numeric" in i for i in issues)


def test_out_of_range_value_flagged():
    rows = [_row(sid) for sid in EXPECTED_IDS]
    rows[2][LABEL_COLUMNS[0]] = "1.5"
    is_valid, issues = validate_submission(rows, HEADER, EXPECTED_IDS)
    assert not is_valid
    assert any("outside [0.0, 1.0]" in i for i in issues)


def test_nan_and_inf_values_flagged():
    rows = [_row(sid) for sid in EXPECTED_IDS]
    rows[0][LABEL_COLUMNS[0]] = "nan"
    rows[1][LABEL_COLUMNS[1]] = "inf"
    is_valid, issues = validate_submission(rows, HEADER, EXPECTED_IDS)
    assert not is_valid
    assert sum("non-finite" in i for i in issues) == 2


def test_boundary_values_0_and_1_are_valid():
    rows = [_row(sid, value=v) for sid, v in zip(EXPECTED_IDS, [0.0, 1.0, 0.5])]
    is_valid, issues = validate_submission(rows, HEADER, EXPECTED_IDS)
    assert is_valid and issues == []


def test_many_value_issues_get_truncated_with_a_summary():
    rows = [_row(sid) for sid in EXPECTED_IDS]
    for row in rows:
        for col in LABEL_COLUMNS:
            row[col] = "9.0"  # every cell out of range -> 3*12 = 36 issues
    is_valid, issues = validate_submission(rows, HEADER, EXPECTED_IDS)
    assert not is_valid
    assert any("more value issues" in i for i in issues)
    assert len(issues) < 36


def test_custom_value_range():
    rows = [_row(sid, value=2.0) for sid in EXPECTED_IDS]
    is_valid, issues = validate_submission(rows, HEADER, EXPECTED_IDS, value_range=(0.0, 10.0))
    assert is_valid and issues == []


# --- file-based helpers, against the real local sample_submission.csv --------

SAMPLE_SUB = Path(__file__).resolve().parents[1] / "data" / "raw" / "sample_submission.csv"
needs_real = __import__("pytest").mark.skipif(not SAMPLE_SUB.exists(), reason="sample_submission.csv not downloaded")


@needs_real
def test_real_sample_submission_validates_against_itself():
    expected_ids = expected_ids_from_csv(SAMPLE_SUB)
    is_valid, issues = validate_submission_file(SAMPLE_SUB, expected_ids)
    assert is_valid, issues


@needs_real
def test_load_submission_csv_matches_expected_columns():
    header, rows = load_submission_csv(SAMPLE_SUB)
    assert header == [ID_COLUMN] + LABEL_COLUMNS
    assert len(rows) == len(expected_ids_from_csv(SAMPLE_SUB))
