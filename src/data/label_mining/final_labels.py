"""Step 6 of the OA weak-supervision pipeline, plus assembly of every label
source into one training label table.

Soft labels from the label model become hard labels only where the model is
confident; the band in between is MASKED (no label) rather than guessed, so
the loss can skip it -- the same "abstain instead of guessing wrong"
philosophy the miners already follow. Provenance is kept per cell so
downstream code can weight gold above mined labels, or exclude the gold
studies from training when they're needed for evaluation.

Label matrices throughout are (n_studies, 12) int8 in LABEL_COLUMNS order
with values 1 (positive), 0 (negative), MASKED (-1, no label).
"""

import numpy as np

from data.label_mining.rule_based import mcl_mentioned, mine_report
from data.label_mining.weak_supervision.oa_lfs import (
    COMPARTMENTS,
    generic_no_compartment_flag,
    has_compartment_mention,
)
from utils.constants import LABEL_COLUMNS

MASKED = -1

SOURCE_NONE = 0
SOURCE_GOLD = 1
SOURCE_RULE = 2
SOURCE_WEAK = 3
SOURCE_UNMENTIONED = 4
SOURCE_NAMES = {
    SOURCE_NONE: "none",
    SOURCE_GOLD: "gold",
    SOURCE_RULE: "rule-based",
    SOURCE_WEAK: "weak-supervision",
    SOURCE_UNMENTIONED: "unmentioned-negative",
}


def threshold_soft_labels(soft, pos_threshold=0.85, neg_threshold=0.15):
    """soft: (n,) array of P(positive). Returns (n,) int8: 1 if soft >
    pos_threshold, 0 if soft < neg_threshold, MASKED otherwise. Items with
    no signal at all (soft == 0.5) always land in the masked band.
    """
    if not 0.0 <= neg_threshold < 0.5 < pos_threshold <= 1.0:
        raise ValueError(
            f"need 0 <= neg_threshold < 0.5 < pos_threshold <= 1, got "
            f"neg={neg_threshold}, pos={pos_threshold}"
        )
    soft = np.asarray(soft, dtype=float)
    out = np.full(soft.shape, MASKED, dtype=np.int8)
    out[soft > pos_threshold] = 1
    out[soft < neg_threshold] = 0
    return out


def rule_label_matrix(reports, columns=LABEL_COLUMNS):
    """(n, len(columns)) int8 from rule_based.mine_report. A miner's
    "no signal" (None) and unimplemented labels both become MASKED.
    """
    out = np.full((len(reports), len(columns)), MASKED, dtype=np.int8)
    for i, text in enumerate(reports):
        mined = mine_report(text)
        for j, col in enumerate(columns):
            value = mined[col].value
            if value is not None:
                out[i, j] = value
    return out


def unmentioned_negative_matrix(reports, columns=LABEL_COLUMNS):
    """(n, len(columns)) int8: 0 where the report never mentions the finding
    at all, MASKED everywhere else. Only for columns where "not mentioned"
    was checked against gold dev (2026-09-24) and is reliably negative:

    - OA compartments: no OA term attributed to the compartment (not even a
      negated/hedged one) AND no unattributed OA mention in the report
      ("Gonarthrosis" alone must stay masked, not become 3 negatives).
      Gold dev: 88% / 94% / 88% negative (Medial / Lateral / PF, n=17-18).
    [redacted: quoted text from a restricted clinical dataset]
      7/7 negative.

    - Synovitis: EVERY cell, as a positive-unlabeled (PU) fill -- the miner
      is almost positive-only (368 pos / 45 neg), and 33% of its unlabeled
      gold-dev studies are positive, so these are NOT trusted negatives:
      label_weights() down-weights them to SYNOVITIS_PU_WEIGHT. Being the
      lowest-priority source, any real rule-based 0/1 still wins.

    ACL, menisci, Effusion were checked too and left out: 25-41% of their
    unlabeled studies are gold-positive.
    """
    out = np.full((len(reports), len(columns)), MASKED, dtype=np.int8)
    out[:, columns.index("Synovitis")] = 0
    mcl_j = columns.index("MCL")
    oa_js = {c: columns.index(c) for c in COMPARTMENTS}
    for i, text in enumerate(reports):
        if not mcl_mentioned(text):
            out[i, mcl_j] = 0
        if generic_no_compartment_flag(text):
            continue
        for c, j in oa_js.items():
            if not has_compartment_mention(text, c):
                out[i, j] = 0
    return out


# Loss weight for Synovitis's PU-filled negatives. Not tunable offline (no
# gold signal for it without a retrain); chosen low because a third of them
# are really positive. Ranking-wise, "labeled-positive vs unlabeled" is
# rank-equivalent to the true classifier if mentions are close to random
# among positives -- the low weight limits damage where that fails.
SYNOVITIS_PU_WEIGHT = 0.3


def label_weights(source, columns=LABEL_COLUMNS):
    """(n, len(columns)) float32 per-cell loss weights: 1.0 everywhere except
    Synovitis cells filled by the unmentioned (PU) source.
    """
    w = np.ones(source.shape, dtype=np.float32)
    j = columns.index("Synovitis")
    w[source[:, j] == SOURCE_UNMENTIONED, j] = SYNOVITIS_PU_WEIGHT
    return w


def assemble_training_labels(gold, rule, weak, unmentioned=None):
    """Combine three same-shape label matrices into (labels, source).

    Priority when several sources label the same cell: gold beats
    weak-supervision beats rule-based beats unmentioned-negative (optional,
    lowest -- it only fills cells no real evidence speaks to). (Rule-based and weak-supervision
    cover disjoint labels today, so that ordering only matters if that
    changes.) source holds a SOURCE_* code per cell.
    """
    if not (gold.shape == rule.shape == weak.shape):
        raise ValueError(f"shape mismatch: {gold.shape}, {rule.shape}, {weak.shape}")
    labels = np.full(gold.shape, MASKED, dtype=np.int8)
    source = np.full(gold.shape, SOURCE_NONE, dtype=np.int8)
    layers = [(SOURCE_RULE, rule), (SOURCE_WEAK, weak), (SOURCE_GOLD, gold)]
    if unmentioned is not None:
        if unmentioned.shape != gold.shape:
            raise ValueError(f"shape mismatch: unmentioned {unmentioned.shape} vs {gold.shape}")
        layers.insert(0, (SOURCE_UNMENTIONED, unmentioned))
    for code, matrix in layers:
        has_label = matrix != MASKED
        labels[has_label] = matrix[has_label]
        source[has_label] = code
    return labels, source


def label_summary(labels, source, columns=LABEL_COLUMNS):
    """Per-column counts: positives, negatives, masked, and which source
    supplied the labeled cells. Returns a list of dicts, one per column.
    """
    rows = []
    for j, col in enumerate(columns):
        col_labels = labels[:, j]
        col_source = source[:, j]
        labeled = col_labels != MASKED
        rows.append({
            "label": col,
            "n_positive": int((col_labels == 1).sum()),
            "n_negative": int((col_labels == 0).sum()),
            "n_masked": int((~labeled).sum()),
            "sources": {
                SOURCE_NAMES[code]: int((col_source == code).sum())
                for code in (SOURCE_GOLD, SOURCE_RULE, SOURCE_WEAK, SOURCE_UNMENTIONED)
                if (col_source == code).any()
            },
        })
    return rows
