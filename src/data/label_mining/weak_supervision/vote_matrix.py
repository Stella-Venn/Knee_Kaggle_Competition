"""Vote matrix construction -- Step 2 of the weak-supervision pipeline
([redacted: private project notes], 2026-09-04 decision entry).

Runs every implemented, vote-contributing OA LF (`oa_lfs.LF_FACTORIES`)
against a set of reports, producing an (n_studies, n_columns) matrix of
votes. Deliberately needs no labels -- that's the actual weak-supervision
trick: Step 5's aggregator learns each LF's reliability from patterns of
agreement/disagreement across this matrix, not from gold labels. Meant to
run over all 4407 training reports (labeled and unlabeled alike), not just
the 58 gold-labeled ones or the 40-study dev subset used to build the LFs.

Also includes the translate-then-match columns (decided 2026-09-04): same
`LF_FACTORIES` functions, applied to pre-translated English text instead of
the original -- see oa_lfs.py's comment on why no separate matching
function was needed, and translate.py for the translation step itself.
Requires a separate `translated_reports` argument since it's genuinely
different input, not a new LF implementation.
"""

import numpy as np

from data.label_mining.weak_supervision.oa_lfs import (
    COMPARTMENTS,
    DISABLED_LF_COLUMNS,
    LF_FACTORIES,
)

TRANSLATE_PREFIX = "translate_"

# Stable column order: (compartment, lf_name) pairs, so matrix[:, j]
# reliably means columns[j] regardless of dict iteration order elsewhere.
# Includes DISABLED_LF_COLUMNS (e.g. Lateral OA's below-chance
# keyword_proximity, per Step 3's audit) -- the raw matrix keeps every LF's
# votes for analysis; ACTIVE_VOTE_COLUMNS below is what Step 5's aggregator
# should actually fit/predict on. A "translate_"-prefixed lf_name (e.g.
# "translate_keyword_proximity") means: run LF_FACTORIES[name without the
# prefix] against translated_reports instead of reports.
VOTE_COLUMNS = [
    (compartment, lf_name)
    for compartment in COMPARTMENTS
    for lf_name in list(LF_FACTORIES) + [TRANSLATE_PREFIX + n for n in LF_FACTORIES]
]
ACTIVE_VOTE_COLUMNS = [c for c in VOTE_COLUMNS if c not in DISABLED_LF_COLUMNS]


def build_vote_matrix(reports, translated_reports=None):
    """Run every OA LF against every report.

    reports: list[str]. translated_reports: list[str] | None, same length/
    order as `reports`, pre-translated to English (see translate.py) --
    required if VOTE_COLUMNS includes any "translate_"-prefixed column
    (it does by default); pass None only if you've filtered VOTE_COLUMNS
    down to non-translate columns yourself.

    Returns (matrix, columns): matrix is an (len(reports), len(VOTE_COLUMNS))
    int array of votes (POSITIVE=1, NEGATIVE=-1, ABSTAIN=0); columns is
    VOTE_COLUMNS. Study IDs are deliberately not part of the matrix -- keep
    them alongside (same row order) when persisting, so a row can be traced
    back to its StudyInstanceUID.
    """
    needs_translation = any(lf.startswith(TRANSLATE_PREFIX) for _, lf in VOTE_COLUMNS)
    if needs_translation and translated_reports is None:
        raise ValueError(
            "VOTE_COLUMNS includes translate_* columns but translated_reports "
            "wasn't provided -- see translate.py's translate_batch."
        )
    if translated_reports is not None and len(translated_reports) != len(reports):
        raise ValueError("reports and translated_reports must be the same length")

    lfs = {}
    for compartment in COMPARTMENTS:
        for lf_name, factory in LF_FACTORIES.items():
            lfs[(compartment, lf_name)] = factory(compartment)
            lfs[(compartment, TRANSLATE_PREFIX + lf_name)] = factory(compartment)

    matrix = np.zeros((len(reports), len(VOTE_COLUMNS)), dtype=int)
    for i in range(len(reports)):
        for j, (compartment, lf_name) in enumerate(VOTE_COLUMNS):
            text = translated_reports[i] if lf_name.startswith(TRANSLATE_PREFIX) else reports[i]
            matrix[i, j] = lfs[(compartment, lf_name)](text)
    return matrix, VOTE_COLUMNS


def vote_matrix_summary(matrix, columns=VOTE_COLUMNS):
    """Per-column coverage (fraction not ABSTAIN) and vote split -- a quick,
    label-free sanity check on the matrix before Step 3's gold-labeled audit.
    """
    n = matrix.shape[0]
    summary = {}
    for j, col in enumerate(columns):
        votes = matrix[:, j]
        n_pos = int((votes == 1).sum())
        n_neg = int((votes == -1).sum())
        n_abstain = int((votes == 0).sum())
        summary[col] = {
            "n_positive": n_pos,
            "n_negative": n_neg,
            "n_abstain": n_abstain,
            "coverage": (n_pos + n_neg) / n if n else None,
        }
    return summary
