"""Unit tests for src/data/label_mining/final_labels.py."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from data.label_mining.final_labels import (
    MASKED,
    SOURCE_GOLD,
    SOURCE_NONE,
    SOURCE_RULE,
    SOURCE_WEAK,
    assemble_training_labels,
    label_summary,
    rule_label_matrix,
    threshold_soft_labels,
)
from utils.constants import LABEL_COLUMNS


def test_threshold_default_bands():
    soft = np.array([0.99, 0.851, 0.85, 0.5, 0.15, 0.149, 0.01])
    out = threshold_soft_labels(soft)
    # boundaries are exclusive: exactly 0.85 and exactly 0.15 are masked
    assert list(out) == [1, 1, MASKED, MASKED, MASKED, 0, 0]


def test_threshold_no_signal_is_masked():
    assert threshold_soft_labels(np.array([0.5]))[0] == MASKED


def test_threshold_custom_values():
    out = threshold_soft_labels(np.array([0.7, 0.3, 0.5]), pos_threshold=0.6, neg_threshold=0.4)
    assert list(out) == [1, 0, MASKED]


@pytest.mark.parametrize("pos,neg", [(0.4, 0.1), (0.9, 0.6), (0.85, 0.5), (1.2, 0.1)])
def test_threshold_rejects_bad_thresholds(pos, neg):
    with pytest.raises(ValueError):
        threshold_soft_labels(np.array([0.5]), pos_threshold=pos, neg_threshold=neg)


def test_rule_label_matrix_shape_and_masking():
    reports = ["No fracture is seen.", "A popliteal cyst of 20mm is seen."]
    out = rule_label_matrix(reports)
    assert out.shape == (2, len(LABEL_COLUMNS))
    assert out.dtype == np.int8
    j_fracture = LABEL_COLUMNS.index("Fracture")
    j_bakers = LABEL_COLUMNS.index("Baker's")
    j_acl = LABEL_COLUMNS.index("ACL")
    assert out[0, j_fracture] == 0
    assert out[1, j_bakers] == 1
    assert out[0, j_acl] == MASKED  # no miner for ACL yet


def _matrix(rows):
    return np.array(rows, dtype=np.int8)


def test_assemble_priority_gold_over_weak_over_rule():
    gold = _matrix([[1, MASKED, MASKED, MASKED]])
    rule = _matrix([[0, 0, MASKED, MASKED]])
    weak = _matrix([[MASKED, 1, 1, MASKED]])
    labels, source = assemble_training_labels(gold, rule, weak)
    assert list(labels[0]) == [1, 1, 1, MASKED]
    assert list(source[0]) == [SOURCE_GOLD, SOURCE_WEAK, SOURCE_WEAK, SOURCE_NONE]


def test_assemble_rule_only_cell():
    gold = _matrix([[MASKED]])
    rule = _matrix([[0]])
    weak = _matrix([[MASKED]])
    labels, source = assemble_training_labels(gold, rule, weak)
    assert labels[0, 0] == 0 and source[0, 0] == SOURCE_RULE


def test_assemble_rejects_shape_mismatch():
    with pytest.raises(ValueError):
        assemble_training_labels(_matrix([[1, 1]]), _matrix([[1]]), _matrix([[1]]))


def test_label_summary_counts():
    labels = _matrix([[1, 0], [0, MASKED], [MASKED, MASKED]])
    source = _matrix([[SOURCE_GOLD, SOURCE_RULE], [SOURCE_RULE, SOURCE_NONE], [SOURCE_NONE, SOURCE_NONE]])
    rows = label_summary(labels, source, columns=["a", "b"])
    a, b = rows
    assert (a["n_positive"], a["n_negative"], a["n_masked"]) == (1, 1, 1)
    assert a["sources"] == {"gold": 1, "rule-based": 1}
    assert (b["n_positive"], b["n_negative"], b["n_masked"]) == (0, 1, 2)
    assert b["sources"] == {"rule-based": 1}


# --- unmentioned-negative source (2026-09-24) -------------------------------

from data.label_mining.final_labels import (  # noqa: E402
    SOURCE_UNMENTIONED,
    unmentioned_negative_matrix,
)


def _col(name):
    return LABEL_COLUMNS.index(name)


def test_unmentioned_fills_oa_and_mcl_only_when_never_mentioned():
    reports = [
        "ACL tear. Medial meniscus fine. Collateral ligaments look intact.",  # no OA at all
        "Medial compartment cartilage: complete cartilage loss. MCL fine.",  # medial OA mentioned
        "Gonarthrosis. No other findings.",  # OA with no compartment -> stay masked
    ]
    m = unmentioned_negative_matrix(reports)
    # report 0: all three OA compartments unmentioned -> 0; MCL mentioned (collateral) -> masked
    assert all(m[0, _col(c)] == 0 for c in ("Medial OA", "Lateral OA", "PF OA"))
    assert m[0, _col("MCL")] == MASKED
    # report 1: medial mentioned -> masked; lateral/PF unmentioned -> 0
    assert m[1, _col("Medial OA")] == MASKED
    assert m[1, _col("Lateral OA")] == 0 and m[1, _col("PF OA")] == 0
    # report 2: unattributed OA mention keeps every compartment masked
    assert all(m[2, _col(c)] == MASKED for c in ("Medial OA", "Lateral OA", "PF OA"))
    assert m[2, _col("MCL")] == 0
    # never touches labels it wasn't validated for
    for c in ("ACL", "Effusion", "Medial Meniscus"):
        assert (m[:, _col(c)] == MASKED).all()
    # Synovitis: PU fill on every row (down-weighted in training, see label_weights)
    assert (m[:, _col("Synovitis")] == 0).all()


def test_label_weights_downweight_only_synovitis_pu_cells():
    from data.label_mining.final_labels import SOURCE_RULE, SYNOVITIS_PU_WEIGHT, label_weights
    source = np.full((2, len(LABEL_COLUMNS)), SOURCE_UNMENTIONED, dtype=np.int8)
    source[1, _col("Synovitis")] = SOURCE_RULE
    w = label_weights(source)
    assert w[0, _col("Synovitis")] == pytest.approx(SYNOVITIS_PU_WEIGHT)
    assert w[1, _col("Synovitis")] == 1.0  # real miner label keeps full weight
    assert w[0, _col("Lateral OA")] == 1.0  # OA/MCL unmentioned fills are trusted (88-100% on dev)


def test_unmentioned_is_lowest_priority_source():
    n = len(LABEL_COLUMNS)
    gold = np.full((1, n), MASKED, dtype=np.int8)
    rule = np.full((1, n), MASKED, dtype=np.int8)
    weak = np.full((1, n), MASKED, dtype=np.int8)
    un = np.zeros((1, n), dtype=np.int8)
    weak[0, _col("Lateral OA")] = 1
    labels, source = assemble_training_labels(gold, rule, weak, un)
    assert labels[0, _col("Lateral OA")] == 1  # real evidence beats the fill
    assert labels[0, _col("Medial OA")] == 0 and source[0, _col("Medial OA")] == SOURCE_UNMENTIONED
    # default (no unmentioned arg) unchanged
    labels2, _ = assemble_training_labels(gold, rule, weak)
    assert labels2[0, _col("Medial OA")] == MASKED
