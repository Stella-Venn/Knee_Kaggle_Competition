"""Submission format validator ([redacted: private project notes] working agreement: "write the
format-check script once ... and run it automatically before every upload
[redacted: detail about the data challenge this code was written for]
no pandas -- consistent with the rest of this project, and this is a small
enough check not to need it.

[redacted: detail about the data challenge this code was written for]
shape than the 3 visible studies in test.csv ([redacted: private project notes], "Data format").
`validate_submission` therefore takes `expected_study_ids` as an explicit
argument rather than assuming any fixed row count -- a real inference-kernel
run (module 9) derives it from whatever test.csv it actually finds at
scoring time, not from this repo's local copy.
"""

import csv
from collections import Counter

from utils.constants import LABEL_COLUMNS

ID_COLUMN = "StudyInstanceUID"
DEFAULT_VALUE_RANGE = (0.0, 1.0)
_MAX_REPORTED_IDS = 5  # cap how many offending IDs a single issue message lists
_MAX_VALUE_ISSUES = 20  # cap how many per-cell value issues get their own message


def load_submission_csv(path):
    """(header: list[str], rows: list[dict]) via csv.DictReader. `header` is
    `None` for an empty file (no header row at all).
    """
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        header = reader.fieldnames
        rows = list(reader)
    return header, rows


def _truncated(items):
    items = sorted(items)
    shown = items[:_MAX_REPORTED_IDS]
    return f"{shown}{'...' if len(items) > _MAX_REPORTED_IDS else ''}"


def validate_submission(
    rows, header, expected_study_ids, columns=LABEL_COLUMNS,
    id_column=ID_COLUMN, value_range=DEFAULT_VALUE_RANGE,
):
    """Checks a submission (already parsed via load_submission_csv, or built
    directly by an inference kernel) against `expected_study_ids` -- the
    StudyInstanceUIDs the actual test set has, in Kaggle's expected order.

    Checks: column names and order, row count, every expected ID present
    exactly once (no missing/extra/duplicate), row order matches (informational
    but checked -- costs nothing, and a naive scorer that zips rather than
    joins would break on a reordered submission), and every label cell is
    present, numeric, finite, and within `value_range`.

    Returns (is_valid: bool, issues: list[str]). Column/row-count/ID problems
    are checked first and don't block also checking values -- multiple
    problems in one run get reported together rather than one at a time
    across repeated validator calls.
    """
    issues = []
    expected_header = [id_column] + list(columns)
    if header != expected_header:
        issues.append(f"column names/order mismatch: expected {expected_header}, got {header}")

    if len(rows) != len(expected_study_ids):
        issues.append(f"row count mismatch: expected {len(expected_study_ids)} rows, got {len(rows)}")

    got_ids = [r.get(id_column) for r in rows]
    got_id_set = set(got_ids)
    expected_id_set = set(expected_study_ids)

    missing = expected_id_set - got_id_set
    if missing:
        issues.append(f"missing {len(missing)} expected {id_column}(s): {_truncated(missing)}")
    extra = got_id_set - expected_id_set
    if extra:
        issues.append(f"{len(extra)} unexpected {id_column}(s) present: {_truncated(extra)}")
    dup_counts = Counter(got_ids)
    dups = [sid for sid, c in dup_counts.items() if c > 1]
    if dups:
        issues.append(f"{len(dups)} duplicate {id_column}(s): {_truncated(dups)}")

    if not missing and not extra and not dups and got_ids != list(expected_study_ids):
        issues.append(f"{id_column} order differs from the expected order (same set, different order)")

    value_cols = [c for c in columns if header and c in header]
    low, high = value_range
    n_value_issues = 0
    for i, row in enumerate(rows):
        for col in value_cols:
            raw = row.get(col)
            problem = None
            if raw is None or raw == "":
                problem = f"row {i} ({row.get(id_column)}): missing value in column {col!r}"
            else:
                try:
                    v = float(raw)
                except ValueError:
                    problem = f"row {i} ({row.get(id_column)}): non-numeric value {raw!r} in column {col!r}"
                else:
                    if v != v or v in (float("inf"), float("-inf")):  # v != v catches NaN
                        problem = f"row {i} ({row.get(id_column)}): non-finite value {v} in column {col!r}"
                    elif not (low <= v <= high):
                        problem = (f"row {i} ({row.get(id_column)}): value {v} in column {col!r} "
                                   f"outside [{low}, {high}]")
            if problem:
                n_value_issues += 1
                if n_value_issues <= _MAX_VALUE_ISSUES:
                    issues.append(problem)
    if n_value_issues > _MAX_VALUE_ISSUES:
        issues.append(f"... and {n_value_issues - _MAX_VALUE_ISSUES} more value issues (truncated)")

    return len(issues) == 0, issues


def validate_submission_file(submission_path, expected_study_ids, columns=LABEL_COLUMNS):
    """Convenience wrapper: reads `submission_path` from disk and validates it."""
    header, rows = load_submission_csv(submission_path)
    return validate_submission(rows, header, expected_study_ids, columns)


def expected_ids_from_csv(path, id_column=ID_COLUMN):
    """StudyInstanceUIDs (in file order) from a test.csv or sample_submission.csv
    -- whichever one reflects the actual test set at hand.
    """
    header, rows = load_submission_csv(path)
    return [r[id_column] for r in rows]
