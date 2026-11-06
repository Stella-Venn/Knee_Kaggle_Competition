"""Unit tests for src/data/label_mining/weak_supervision/label_model.py."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from data.label_mining.weak_supervision.label_model import fit_label_model
from data.label_mining.weak_supervision.oa_lfs import COMPARTMENTS
from data.label_mining.weak_supervision.vote_matrix import VOTE_COLUMNS
from data.label_mining.weak_supervision.label_model import fit_oa_label_models

# Two LFs that fully agree establish a strong, self-reinforcing consensus
# (a single 2-LF setup is degenerate for this -- see
# test_two_perfectly_opposed_lfs_are_symmetric_ambiguity below for why a
# third LF is needed to break the tie). A third, alternating LF is
# inconsistent with that consensus about half the time. No gold labels
# involved anywhere -- only agreement patterns, matching the real algorithm.
_LF_A = [1] * 15 + [-1] * 15
_LF_B = [1] * 15 + [-1] * 15
_LF_NOISY = [1, -1] * 7 + [1] + [-1, 1] * 7 + [-1]


def test_reliable_lf_gets_higher_alpha_than_noisy_lf():
    votes = np.array([_LF_A, _LF_B, _LF_NOISY]).T
    p, alphas = fit_label_model(votes)
    assert alphas[0] > alphas[2]
    assert alphas[1] > alphas[2]
    assert alphas[0] > 0.9  # the reliable, agreeing LFs should dominate
    assert 0.4 < alphas[2] < 0.65  # the noisy LF should end up near chance


def test_soft_labels_track_the_reliable_lfs():
    votes = np.array([_LF_A, _LF_B, _LF_NOISY]).T
    p, _ = fit_label_model(votes)
    assert p[:15].mean() > 0.9
    assert p[15:].mean() < 0.1


def test_two_perfectly_opposed_lfs_are_symmetric_ambiguity():
    # With only 2 LFs that perfectly disagree on every item, there's no
    # signal to break the tie between "LF1 is right" and "LF2 is right" --
    # both ending up at alpha=0.5 is the mathematically correct result for
    # this degenerate case, not a bug. Documents why the other tests here
    # use 3 LFs instead of 2.
    lf_reliable = [1] * 15 + [-1] * 15
    lf_anti = [-1] * 15 + [1] * 15
    votes = np.array([lf_reliable, lf_anti]).T
    _, alphas = fit_label_model(votes)
    assert alphas[0] == alphas[1] == 0.5


def test_item_with_no_votes_gets_no_signal():
    votes = np.array([[0, 0], [1, 1]])
    p, _ = fit_label_model(votes)
    assert p[0] == 0.5


def test_no_lfs_returns_uninformative_prior():
    votes = np.zeros((5, 0), dtype=int)
    p, alphas = fit_label_model(votes)
    assert (p == 0.5).all()
    assert alphas.shape == (0,)


def test_anti_correlated_lf_gets_low_alpha():
    # An LF that always disagrees with an established consensus (two
    # agreeing LFs, breaking the 2-LF symmetry problem above) should end up
    # recognized as actively unreliable -- alpha near the clip floor, not
    # just "not helpful" -- rather than requiring a manual kill decision the
    # way Step 3 needed for Lateral OA's keyword_proximity.
    lf_anti = [-1] * 15 + [1] * 15  # perfectly opposite of _LF_A/_LF_B
    votes = np.array([_LF_A, _LF_B, lf_anti]).T
    p, alphas = fit_label_model(votes)
    assert alphas[2] < alphas[0]
    assert alphas[2] < 0.1  # recognized as strongly anti-correlated, not just weak


def test_perfect_agreement_gives_confident_soft_labels():
    lf_a = [1, 1, -1, -1]
    lf_b = [1, 1, -1, -1]  # identical to lf_a -- full agreement throughout
    votes = np.array([lf_a, lf_b]).T
    p, alphas = fit_label_model(votes)
    assert p[0] > 0.9 and p[1] > 0.9
    assert p[2] < 0.1 and p[3] < 0.1


def test_fit_oa_label_models_returns_all_compartments():
    n_items = 5
    matrix = np.zeros((n_items, len(VOTE_COLUMNS)), dtype=int)
    results = fit_oa_label_models(matrix, VOTE_COLUMNS)
    assert set(results.keys()) == set(COMPARTMENTS)
    for compartment, r in results.items():
        assert r["soft_labels"].shape == (n_items,)
        assert isinstance(r["lf_reliability"], dict)


def test_fit_oa_label_models_all_abstain_gives_no_signal():
    n_items = 3
    matrix = np.zeros((n_items, len(VOTE_COLUMNS)), dtype=int)
    results = fit_oa_label_models(matrix, VOTE_COLUMNS)
    for r in results.values():
        assert (r["soft_labels"] == 0.5).all()


# --- fixed_alphas / alpha_overrides (2026-09-23) ----------------------------

def test_fixed_alphas_default_none_matches_unfixed_behavior():
    votes = np.array([_LF_A, _LF_B, _LF_NOISY]).T
    p1, a1 = fit_label_model(votes)
    p2, a2 = fit_label_model(votes, fixed_alphas=None)
    np.testing.assert_array_equal(p1, p2)
    np.testing.assert_array_equal(a1, a2)


def test_fixed_alphas_holds_specified_lf_constant():
    # Force the noisy LF's alpha to 0.9 (as if a gold audit said it's actually
    # reliable) -- it should stay exactly 0.9 across all iterations, not drift
    # back toward what unsupervised agreement would have estimated for it.
    votes = np.array([_LF_A, _LF_B, _LF_NOISY]).T
    fixed = np.array([np.nan, np.nan, 0.9])
    _, alphas = fit_label_model(votes, fixed_alphas=fixed)
    assert alphas[2] == pytest.approx(0.9)


def test_fixed_alphas_still_lets_other_lfs_update_normally():
    votes = np.array([_LF_A, _LF_B, _LF_NOISY]).T
    _, alphas_free = fit_label_model(votes)
    fixed = np.array([np.nan, np.nan, 0.9])
    _, alphas_fixed = fit_label_model(votes, fixed_alphas=fixed)
    # LF_A/LF_B (not fixed) should end up close to their free-fit values --
    # not identical, since a differently-weighted LF_NOISY changes the
    # consensus they're measured against, but not wildly different either.
    assert alphas_fixed[0] == pytest.approx(alphas_free[0], abs=0.15)
    assert alphas_fixed[1] == pytest.approx(alphas_free[1], abs=0.15)


def test_fixed_alphas_overriding_toward_low_reliability_flips_soft_labels():
    # If a high-coverage LF's votes are wrong more often than an unsupervised
    # fit would assume, forcing its alpha down (as gold evidence would
    # demand) should measurably shift the resulting soft labels away from
    # blindly trusting it.
    lf_dominant_wrong = [1] * 20  # votes positive on everything
    lf_reliable = [-1] * 15 + [1] * 5  # actually mostly negative
    votes = np.array([lf_dominant_wrong, lf_reliable]).T
    p_free, _ = fit_label_model(votes)
    fixed = np.array([0.3, np.nan])  # tell it lf_dominant_wrong is actually unreliable
    p_fixed, alphas_fixed = fit_label_model(votes, fixed_alphas=fixed)
    assert alphas_fixed[0] == pytest.approx(0.3)
    # with lf_dominant_wrong down-weighted, the mostly-negative items (0-14)
    # should trend lower toward lf_reliable's (correct) negative vote
    assert p_fixed[:15].mean() < p_free[:15].mean()


def test_fixed_alphas_rejects_wrong_shape():
    votes = np.array([_LF_A, _LF_B]).T
    with pytest.raises(ValueError):
        fit_label_model(votes, fixed_alphas=np.array([0.5]))


def test_fit_oa_label_models_alpha_overrides_applied_to_matching_lf():
    n_items = 20
    matrix = np.zeros((n_items, len(VOTE_COLUMNS)), dtype=int)
    kp_idx = VOTE_COLUMNS.index(("Lateral OA", "keyword_proximity"))
    neg_idx = VOTE_COLUMNS.index(("Lateral OA", "negation"))
    matrix[:10, kp_idx] = 1
    matrix[10:, kp_idx] = -1
    matrix[:10, neg_idx] = 1
    matrix[10:, neg_idx] = -1

    overrides = {("Lateral OA", "keyword_proximity"): 0.4}
    results = fit_oa_label_models(matrix, VOTE_COLUMNS, alpha_overrides=overrides)
    assert results["Lateral OA"]["lf_reliability"]["keyword_proximity"] == pytest.approx(0.4)
    # other compartments/LFs untouched by an override that names a different compartment
    assert results["Medial OA"]["lf_reliability"]["keyword_proximity"] != pytest.approx(0.4)


def test_fit_oa_label_models_alpha_overrides_none_matches_default():
    n_items = 5
    matrix = np.zeros((n_items, len(VOTE_COLUMNS)), dtype=int)
    r1 = fit_oa_label_models(matrix, VOTE_COLUMNS)
    r2 = fit_oa_label_models(matrix, VOTE_COLUMNS, alpha_overrides=None)
    for compartment in r1:
        np.testing.assert_array_equal(r1[compartment]["soft_labels"], r2[compartment]["soft_labels"])
        assert r1[compartment]["lf_reliability"] == r2[compartment]["lf_reliability"]
