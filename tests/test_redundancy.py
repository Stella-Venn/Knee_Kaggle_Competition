"""Unit tests for src/data/label_mining/weak_supervision/redundancy.py.

Uses tiny hand-constructed vote matrices with known agreement by
construction, not real data -- the real-data numbers are checked
separately via scripts/check_lf_redundancy.py.
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from data.label_mining.weak_supervision.oa_lfs import ABSTAIN, NEGATIVE, POSITIVE
from data.label_mining.weak_supervision.redundancy import (
    MIN_COVERAGE_TO_JUDGE,
    merge_translate_duplicates,
    pairwise_agreement,
)
from data.label_mining.weak_supervision.vote_matrix import VOTE_COLUMNS


def _col(compartment, lf_name):
    return VOTE_COLUMNS.index((compartment, lf_name))


def _find(results, compartment, lf_a, lf_b):
    for r in results:
        if r["compartment"] == compartment and {r["lf_a"], r["lf_b"]} == {lf_a, lf_b}:
            return r
    raise KeyError((compartment, lf_a, lf_b))


def test_identical_columns_are_fully_redundant():
    n = 20
    matrix = np.zeros((n, len(VOTE_COLUMNS)), dtype=int)
    votes = np.array([1, -1, 1, -1] * 5)
    matrix[:, _col("Medial OA", "keyword_proximity")] = votes
    matrix[:, _col("Medial OA", "translate_keyword_proximity")] = votes  # identical

    results = pairwise_agreement(matrix)
    r = _find(results, "Medial OA", "keyword_proximity", "translate_keyword_proximity")
    assert r["n_both_fired"] == n
    assert r["agreement"] == 1.0
    assert r["verdict"] == "redundant"


def test_perfectly_opposite_columns_are_independent():
    n = 20
    matrix = np.zeros((n, len(VOTE_COLUMNS)), dtype=int)
    votes_a = np.array([1, -1] * 10)
    votes_b = -votes_a  # always disagree when both fire
    matrix[:, _col("Lateral OA", "keyword_proximity")] = votes_a
    matrix[:, _col("Lateral OA", "translate_keyword_proximity")] = votes_b

    results = pairwise_agreement(matrix)
    r = _find(results, "Lateral OA", "keyword_proximity", "translate_keyword_proximity")
    assert r["n_both_fired"] == n
    assert r["agreement"] == 0.0
    assert r["verdict"] == "independent"


def test_few_jointly_firing_rows_is_insufficient_data():
    n = 100
    matrix = np.zeros((n, len(VOTE_COLUMNS)), dtype=int)
    # Only fire together on a handful of rows, well under MIN_COVERAGE_TO_JUDGE.
    n_joint = MIN_COVERAGE_TO_JUDGE - 1
    matrix[:n_joint, _col("PF OA", "keyword_proximity")] = 1
    matrix[:n_joint, _col("PF OA", "negation")] = -1  # disagreeing is irrelevant here

    results = pairwise_agreement(matrix)
    r = _find(results, "PF OA", "keyword_proximity", "negation")
    assert r["n_both_fired"] == n_joint
    assert r["verdict"] == "insufficient data"


def test_no_overlap_gives_none_agreement():
    n = 10
    matrix = np.zeros((n, len(VOTE_COLUMNS)), dtype=int)
    matrix[:5, _col("Medial OA", "keyword_proximity")] = 1
    matrix[5:, _col("Medial OA", "negation")] = -1  # disjoint firing, never both

    results = pairwise_agreement(matrix)
    r = _find(results, "Medial OA", "keyword_proximity", "negation")
    assert r["n_both_fired"] == 0
    assert r["agreement"] is None
    assert r["verdict"] == "insufficient data"


def test_partial_agreement_computed_correctly():
    n = 20
    matrix = np.zeros((n, len(VOTE_COLUMNS)), dtype=int)
    votes_a = np.array([1] * 20)
    votes_b = np.array([1] * 15 + [-1] * 5)  # agree on 15/20
    matrix[:, _col("PF OA", "keyword_proximity")] = votes_a
    matrix[:, _col("PF OA", "translate_keyword_proximity")] = votes_b

    results = pairwise_agreement(matrix)
    r = _find(results, "PF OA", "keyword_proximity", "translate_keyword_proximity")
    assert r["n_both_fired"] == 20
    assert abs(r["agreement"] - 15 / 20) < 1e-9
    assert r["verdict"] == "independent"  # 0.75 < 0.90 threshold


def test_only_within_compartment_pairs_are_computed():
    matrix = np.zeros((10, len(VOTE_COLUMNS)), dtype=int)
    results = pairwise_agreement(matrix)
    for r in results:
        # every pair must share the same compartment by construction --
        # cross-compartment comparisons would be a different question
        # (do two different tasks correlate), not redundancy.
        assert r["lf_a"] != r["lf_b"]
    compartments_seen = {r["compartment"] for r in results}
    assert compartments_seen == {"Medial OA", "Lateral OA", "PF OA"}
    # 4 LFs per compartment -> C(4,2) = 6 pairs each, 3 compartments -> 18 total
    assert len(results) == 18


def test_coverage_overlap_fraction():
    n = 40
    matrix = np.zeros((n, len(VOTE_COLUMNS)), dtype=int)
    matrix[:8, _col("Medial OA", "keyword_proximity")] = 1
    matrix[:8, _col("Medial OA", "translate_keyword_proximity")] = 1

    results = pairwise_agreement(matrix)
    r = _find(results, "Medial OA", "keyword_proximity", "translate_keyword_proximity")
    assert abs(r["coverage_overlap"] - 8 / 40) < 1e-9


def test_merge_prefers_original_when_it_fires():
    matrix = np.zeros((3, len(VOTE_COLUMNS)), dtype=int)
    matrix[0, _col("Medial OA", "keyword_proximity")] = POSITIVE
    matrix[0, _col("Medial OA", "translate_keyword_proximity")] = NEGATIVE  # ignored

    merged_matrix, merged_columns = merge_translate_duplicates(matrix)
    j = merged_columns.index(("Medial OA", "keyword_proximity"))
    assert merged_matrix[0, j] == POSITIVE


def test_merge_falls_back_to_translated_when_original_abstains():
    matrix = np.zeros((3, len(VOTE_COLUMNS)), dtype=int)
    matrix[0, _col("Lateral OA", "negation")] = ABSTAIN
    matrix[0, _col("Lateral OA", "translate_negation")] = NEGATIVE

    merged_matrix, merged_columns = merge_translate_duplicates(matrix)
    j = merged_columns.index(("Lateral OA", "negation"))
    assert merged_matrix[0, j] == NEGATIVE


def test_merge_abstains_when_both_abstain():
    matrix = np.zeros((3, len(VOTE_COLUMNS)), dtype=int)
    merged_matrix, merged_columns = merge_translate_duplicates(matrix)
    j = merged_columns.index(("PF OA", "keyword_proximity"))
    assert merged_matrix[0, j] == ABSTAIN


def test_merge_output_shape_and_columns():
    n = 10
    matrix = np.zeros((n, len(VOTE_COLUMNS)), dtype=int)
    merged_matrix, merged_columns = merge_translate_duplicates(matrix)
    # 3 compartments x 2 merge groups (keyword_proximity, negation)
    assert merged_matrix.shape == (n, 6)
    assert len(merged_columns) == 6
    # no "translate_" prefixed names should survive the merge
    assert all(not lf.startswith("translate_") for _, lf in merged_columns)
    assert ("Medial OA", "keyword_proximity") in merged_columns
    assert ("Lateral OA", "negation") in merged_columns
    assert ("PF OA", "keyword_proximity") in merged_columns


def test_merge_uses_custom_columns_argument():
    # Same VOTE_COLUMNS order but passed explicitly rather than relying on
    # the default, to confirm the columns argument is actually honored.
    matrix = np.zeros((5, len(VOTE_COLUMNS)), dtype=int)
    matrix[0, _col("Medial OA", "keyword_proximity")] = POSITIVE
    merged_matrix, merged_columns = merge_translate_duplicates(matrix, list(VOTE_COLUMNS))
    j = merged_columns.index(("Medial OA", "keyword_proximity"))
    assert merged_matrix[0, j] == POSITIVE
