"""Unit tests for src/data/label_mining/weak_supervision/vote_matrix.py."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from data.label_mining.weak_supervision.oa_lfs import ABSTAIN, DISABLED_LF_COLUMNS, NEGATIVE, POSITIVE
from data.label_mining.weak_supervision.vote_matrix import (
    ACTIVE_VOTE_COLUMNS,
    VOTE_COLUMNS,
    build_vote_matrix,
    vote_matrix_summary,
)

# Synthetic fixture (real report text is never committed). English throughout, so
# "translated" text is identical to the original (translate.py passes English through
# unchanged) -- these tests exercise the vote-matrix plumbing, not translation quality.
_REPORT_ALL_NEGATIVE = (
    " In the medial compartment, the meniscus looks intact on all sequences and "
    "there is no focal chondrosis or chondral damage. \n  \n The lateral compartment, "
    "a complex tear with radial and vertical parts involves the rear horn of the "
    "lateral meniscus. Again no focal chondrosis or chondral damage. \n  \n The "
    "patellofemoral joint, patella and trochlea look smooth without "
    "chondral lesion. The MPFL looks fine."
)
_NO_OA_MENTION = "ACL looks fine. Negative for fracture. Normal alignment."


def test_vote_columns_cover_all_compartments_and_lfs():
    compartments = {c for c, _ in VOTE_COLUMNS}
    lf_names = {name for _, name in VOTE_COLUMNS}
    assert compartments == {"Medial OA", "Lateral OA", "PF OA"}
    assert lf_names == {
        "keyword_proximity", "negation",
        "translate_keyword_proximity", "translate_negation",
    }
    assert len(VOTE_COLUMNS) == 12  # 3 compartments x 4 LF variants


def test_matrix_shape():
    reports = [_REPORT_ALL_NEGATIVE, _NO_OA_MENTION]
    matrix, columns = build_vote_matrix(reports, translated_reports=reports)
    assert matrix.shape == (2, len(VOTE_COLUMNS))
    assert columns == VOTE_COLUMNS


def test_matrix_values_match_direct_lf_calls():
    reports = [_REPORT_ALL_NEGATIVE]
    matrix, columns = build_vote_matrix(reports, translated_reports=reports)
    row = matrix[0]
    for j, (compartment, lf_name) in enumerate(columns):
        if lf_name in ("keyword_proximity", "translate_keyword_proximity"):
            assert row[j] == NEGATIVE, f"{compartment}/{lf_name} expected NEGATIVE"


def test_translate_columns_require_translated_reports():
    with pytest.raises(ValueError):
        build_vote_matrix([_NO_OA_MENTION])  # no translated_reports given


def test_mismatched_lengths_raise():
    with pytest.raises(ValueError):
        build_vote_matrix([_NO_OA_MENTION, _NO_OA_MENTION], translated_reports=[_NO_OA_MENTION])


def test_empty_report_abstains_everywhere():
    matrix, _ = build_vote_matrix([_NO_OA_MENTION], translated_reports=[_NO_OA_MENTION])
    assert (matrix == ABSTAIN).all()


def test_summary_matches_matrix():
    reports = [_REPORT_ALL_NEGATIVE, _NO_OA_MENTION, _REPORT_ALL_NEGATIVE]
    matrix, columns = build_vote_matrix(reports, translated_reports=reports)
    summary = vote_matrix_summary(matrix, columns)

    medial_kw = summary[("Medial OA", "keyword_proximity")]
    assert medial_kw["n_negative"] == 2  # two copies of the all-negative study
    assert medial_kw["n_positive"] == 0
    assert medial_kw["n_abstain"] == 1
    assert abs(medial_kw["coverage"] - 2 / 3) < 1e-9


def test_summary_handles_empty_matrix():
    matrix, columns = build_vote_matrix([], translated_reports=[])
    summary = vote_matrix_summary(matrix, columns)
    for col_summary in summary.values():
        assert col_summary["coverage"] is None


def test_active_vote_columns_excludes_disabled():
    # ACTIVE_VOTE_COLUMNS = VOTE_COLUMNS minus whatever's currently in
    # DISABLED_LF_COLUMNS -- the mechanism, not a specific column, is what's
    # under test (Lateral OA's keyword_proximity was killed here after the
    # original Step 3 audit, then un-killed once real bugs behind its
    # below-chance score were found and fixed -- see oa_lfs.py's
    # DISABLED_LF_COLUMNS comment for the full history).
    assert len(ACTIVE_VOTE_COLUMNS) == len(VOTE_COLUMNS) - len(DISABLED_LF_COLUMNS)
    assert all(c not in ACTIVE_VOTE_COLUMNS for c in DISABLED_LF_COLUMNS)
    assert all(c in VOTE_COLUMNS for c in ACTIVE_VOTE_COLUMNS)
