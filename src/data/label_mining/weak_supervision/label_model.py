"""Step 5 of the OA weak-supervision pipeline: fit a hand-rolled label model
over the vote matrix ([redacted: private project notes], 2026-09-04 decision entry -- hand-
rolled chosen over the `snorkel` package for dependency-friction reasons).

Simplified Dawid-Skene / weighted-majority EM: each LF gets a single
learned reliability scalar (not a full per-LF confusion matrix -- a
deliberate simplification given only 2 LFs per compartment right now, one
of which, `negation`, only ever votes -1 or ABSTAIN by construction; a
single scalar handles that asymmetry naturally without special-casing,
since a direction an LF never emits simply never contributes for it).

No gold labels are used to fit this -- that's the actual weak-supervision
trick. Reliability is learned purely from how well each LF's votes agree
with the current best-guess soft label, iterated via EM. An LF that's
anti-correlated with the consensus (reliability < 0.5) isn't excluded, it's
naturally down-weighted or even inverted by the log-odds math -- log(a/(1-a))
is negative when a < 0.5, so the model learns to distrust or invert it
rather than requiring a manual kill decision the way Step 3 needed one.
"""

import numpy as np

from data.label_mining.weak_supervision.oa_lfs import COMPARTMENTS
from data.label_mining.weak_supervision.vote_matrix import ACTIVE_VOTE_COLUMNS


def fit_label_model(votes, n_iters=10, eps=1e-6, alpha_clip=(0.02, 0.98), fixed_alphas=None):
    """votes: (n_items, n_lfs) int array of {-1, 0, +1} (ABSTAIN=0).

    Returns (soft_labels, alphas): soft_labels is (n_items,) float array of
    P(true label = 1) in [0, 1]; alphas is (n_lfs,) float array of learned
    per-LF reliability (mean agreement with the fitted soft label, among
    that LF's non-abstaining votes). An item with no non-abstaining votes
    at all gets soft_label 0.5 -- no signal, not a guess, consistent with
    how the rule-based miners already treat "no evidence found."

    fixed_alphas: optional (n_lfs,) array-like, `np.nan` for "let EM estimate
    this LF's reliability as usual" and a real value in (0, 1) to hold that
    LF's alpha fixed at across every iteration instead. Needed because
    self-referential agreement-with-consensus can rank an LF's reliability
    backwards from its true, gold-measured accuracy when LFs have very
    unequal coverage -- found for Lateral OA's `keyword_proximity` (30%
    coverage, 58% gold accuracy) vs `negation` (10% coverage, 75% gold
    accuracy): unsupervised EM ranked keyword_proximity as MORE reliable,
    because it dominates the consensus it's also measured against, not
    because it's actually more accurate ([redacted: private project notes], 2026-09-23). A
    fixed LF's votes still feed the log-odds sum that produces soft_labels,
    it just doesn't get to re-estimate its OWN reliability from a consensus
    it helped form.
    """
    n_items, n_lfs = votes.shape
    if n_lfs == 0:
        return np.full(n_items, 0.5), np.zeros(0)

    if fixed_alphas is not None:
        fixed_alphas = np.asarray(fixed_alphas, dtype=float)
        if fixed_alphas.shape != (n_lfs,):
            raise ValueError(f"fixed_alphas shape {fixed_alphas.shape} != ({n_lfs},)")
        fixed_mask = ~np.isnan(fixed_alphas)
    else:
        fixed_mask = np.zeros(n_lfs, dtype=bool)

    vote_sum = votes.sum(axis=1).astype(float)
    n_nonabstain = (votes != 0).sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        p = np.where(
            n_nonabstain > 0,
            (vote_sum / np.maximum(n_nonabstain, 1) + 1) / 2,
            0.5,
        )

    alphas = np.full(n_lfs, 0.5)
    if fixed_mask.any():
        alphas[fixed_mask] = np.clip(fixed_alphas[fixed_mask], *alpha_clip)

    for _ in range(n_iters):
        new_alphas = np.empty(n_lfs)
        for j in range(n_lfs):
            if fixed_mask[j]:
                new_alphas[j] = alphas[j]  # held constant, not re-estimated from consensus agreement
                continue
            mask = votes[:, j] != 0
            if not mask.any():
                new_alphas[j] = 0.5
                continue
            agree = np.where(votes[mask, j] == 1, p[mask], 1 - p[mask])
            new_alphas[j] = agree.mean()
        new_alphas = np.clip(new_alphas, *alpha_clip)

        log_odds_weights = np.log(new_alphas / (1 - new_alphas))
        log_odds = (votes * log_odds_weights).sum(axis=1)
        new_p = 1 / (1 + np.exp(-log_odds))

        converged = np.max(np.abs(new_p - p)) < eps
        p, alphas = new_p, new_alphas
        if converged:
            break

    return p, alphas


def fit_oa_label_models(matrix, columns, n_iters=10, alpha_overrides=None):
    """Fit one label model per OA compartment, using only ACTIVE_VOTE_COLUMNS
    (Step 3's kill decisions respected here, not re-applied by the caller).

    matrix: (n_items, len(columns)) vote matrix from vote_matrix.build_vote_matrix.
    columns: the column list matching matrix's column order (normally
    vote_matrix.VOTE_COLUMNS).

    alpha_overrides: optional dict[(compartment, lf_name)] -> alpha, passed
    through as `fixed_alphas` to `fit_label_model` for the matching LF --
    see that function's docstring for why this is needed (an LF whose true
    reliability is already known from a gold-labeled audit, where letting
    unsupervised EM re-derive it from consensus agreement can get the
    ranking backwards).

    Returns dict[compartment] -> {"soft_labels": (n_items,) array,
    "lf_reliability": {lf_name: alpha}}.
    """
    results = {}
    for compartment in COMPARTMENTS:
        active_idxs = [
            i for i, c in enumerate(columns)
            if c[0] == compartment and c in ACTIVE_VOTE_COLUMNS
        ]
        sub_votes = matrix[:, active_idxs]
        fixed_alphas = None
        if alpha_overrides:
            fixed_alphas = np.full(len(active_idxs), np.nan)
            for k, idx in enumerate(active_idxs):
                key = (compartment, columns[idx][1])
                if key in alpha_overrides:
                    fixed_alphas[k] = alpha_overrides[key]
        p, alphas = fit_label_model(sub_votes, n_iters=n_iters, fixed_alphas=fixed_alphas)
        lf_reliability = {
            columns[idx][1]: float(a) for idx, a in zip(active_idxs, alphas)
        }
        results[compartment] = {"soft_labels": p, "lf_reliability": lf_reliability}
    return results
