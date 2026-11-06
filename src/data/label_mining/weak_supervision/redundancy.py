"""Step 4 of the OA weak-supervision pipeline: pairwise LF redundancy check
([redacted: private project notes], 2026-09-09 decision entry).

Dawid-Skene-style aggregation (Step 5) assumes LFs are roughly conditionally
independent given the true label. Two LFs that mostly restate the same
underlying signal violate that and bias the aggregator toward whatever they
happen to agree on -- exactly the concern flagged when `keyword_proximity`
and `translate_keyword_proximity` converged to the identical reliability
ceiling (alpha=0.98) in every compartment during Step 5's rerun on the real
4407-report pool. This checks that directly with a label-free pairwise
agreement matrix (no gold needed, same as Step 2's vote matrix) rather than
just inferring redundancy from a suspicious-looking alpha.

Only within-compartment pairs are meaningful. Comparing a Medial OA LF's
votes to a Lateral OA LF's votes answers a different question (do these two
DIFFERENT tasks happen to correlate) than redundancy (are these two LFs
restating the same opinion about the SAME task) -- cross-compartment pairs
are deliberately not computed.
"""

from itertools import combinations

import numpy as np

from data.label_mining.weak_supervision.oa_lfs import COMPARTMENTS
from data.label_mining.weak_supervision.vote_matrix import VOTE_COLUMNS

# Thresholds are a starting point, not empirically tuned -- flagged as
# "redundant" only when there's both enough joint coverage to trust the
# number AND agreement high enough to suggest restating rather than
# independently corroborating.
MIN_COVERAGE_TO_JUDGE = 15  # n_both_fired below this: "insufficient data"
REDUNDANT_AGREEMENT_THRESHOLD = 0.90


def pairwise_agreement(matrix, columns=VOTE_COLUMNS):
    """Returns one dict per within-compartment LF pair:
    {compartment, lf_a, lf_b, n_both_fired, coverage_overlap, agreement,
    verdict}.

    agreement: fraction of rows where both LFs fired (non-ABSTAIN) that had
    the same vote value -- None if n_both_fired is 0.
    coverage_overlap: n_both_fired / n_total rows in the matrix.
    verdict: "redundant" (high agreement, enough data to trust it),
    "independent" (enough data, agreement below threshold), or
    "insufficient data" (too few jointly-firing rows to judge either way).
    """
    n_total = matrix.shape[0]
    results = []
    for compartment in COMPARTMENTS:
        compartment_cols = [
            (j, lf_name) for j, (c, lf_name) in enumerate(columns) if c == compartment
        ]
        for (j_a, lf_a), (j_b, lf_b) in combinations(compartment_cols, 2):
            votes_a = matrix[:, j_a]
            votes_b = matrix[:, j_b]
            both_fired = (votes_a != 0) & (votes_b != 0)
            n_both = int(both_fired.sum())
            if n_both == 0:
                agreement = None
            else:
                agreement = float((votes_a[both_fired] == votes_b[both_fired]).mean())

            if n_both < MIN_COVERAGE_TO_JUDGE:
                verdict = "insufficient data"
            elif agreement >= REDUNDANT_AGREEMENT_THRESHOLD:
                verdict = "redundant"
            else:
                verdict = "independent"

            results.append({
                "compartment": compartment,
                "lf_a": lf_a,
                "lf_b": lf_b,
                "n_both_fired": n_both,
                "coverage_overlap": n_both / n_total if n_total else None,
                "agreement": agreement,
                "verdict": verdict,
            })
    return results


# Confirmed via pairwise_agreement() against the real 4407-report pool
# ([redacted: private project notes], 2026-09-09): keyword_proximity/translate_keyword_proximity
# and negation/translate_negation each agree 99-100% of the time they both
# fire, in every compartment -- same decision logic, different input
# language, not independent corroboration. The other four within-compartment
# pairs (any keyword-type LF vs. any negation-type LF) are genuinely
# independent (55.9-88.0% agreement) -- the redundancy is specifically about
# logic, not language, so only these two groups get merged.
MERGE_GROUPS = ["keyword_proximity", "negation"]


def merge_translate_duplicates(matrix, columns=VOTE_COLUMNS):
    """Collapse each confirmed-redundant pair (original-language LF +
    its translate_-prefixed twin) into one column per compartment, so
    Step 5's aggregator sees one opinion per underlying check instead of
    two correlated copies of it.

    Merge rule: prefer the original-language vote; fall back to the
    translated vote only when the original abstains. This keeps whatever
    extra coverage translation provides (cases the original text missed)
    without double-counting the ~99-100% of cases where they'd agree
    anyway.

    Returns (merged_matrix, merged_columns) -- merged_columns uses the
    original (non-"translate_"-prefixed) column names, since the merged
    column represents that same logical check, just with broader coverage.
    """
    merged_columns = []
    merged_cols_data = []
    for compartment in COMPARTMENTS:
        for base_name in MERGE_GROUPS:
            orig_idx = columns.index((compartment, base_name))
            trans_idx = columns.index((compartment, f"translate_{base_name}"))
            orig_votes = matrix[:, orig_idx]
            trans_votes = matrix[:, trans_idx]
            merged_votes = np.where(orig_votes != 0, orig_votes, trans_votes)
            merged_columns.append((compartment, base_name))
            merged_cols_data.append(merged_votes)
    merged_matrix = np.stack(merged_cols_data, axis=1)
    return merged_matrix, merged_columns
