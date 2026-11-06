"""Unit tests for src/data/label_mining/weak_supervision/audit.py."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from data.label_mining.weak_supervision.audit import audit_all, audit_lf

# Synthetic fixtures (real report text is never committed).
_MEDIAL_POSITIVE = (
    "MEDIAL COMPARTMENT:\nMedial compartment cartilage: Full thickness "
    "cartilage loss over the medial femoral condyle and the medial tibial "
    "plateau with subchondral cysts."
)
_MEDIAL_NEGATIVE = (
    "In the medial compartment, the meniscus looks intact and there is no "
    "focal chondrosis or chondral damage."
)
_NO_MENTION = "ACL is intact. No fracture is seen."


def test_audit_lf_counts_true_positive():
    result = audit_lf("Medial OA", "keyword_proximity", [_MEDIAL_POSITIVE], [1])
    assert result["tp"] == 1
    assert result["n_scored"] == 1
    assert result["coverage"] == 1.0
    assert result["accuracy"] == 1.0


def test_audit_lf_counts_true_negative():
    result = audit_lf("Medial OA", "negation", [_MEDIAL_NEGATIVE], [0])
    assert result["tn"] == 1
    assert result["accuracy"] == 1.0


def test_audit_lf_abstain_not_scored():
    result = audit_lf("Medial OA", "keyword_proximity", [_NO_MENTION], [0])
    assert result["n_scored"] == 0
    assert result["coverage"] == 0.0
    assert result["accuracy"] is None


def test_audit_lf_mixed_batch():
    reports = [_MEDIAL_POSITIVE, _MEDIAL_NEGATIVE, _NO_MENTION]
    gold = [1, 0, 0]
    result = audit_lf("Medial OA", "keyword_proximity", reports, gold)
    assert result["n_total"] == 3
    assert result["tp"] == 1  # _MEDIAL_POSITIVE
    assert result["tn"] == 1  # _MEDIAL_NEGATIVE (keyword_proximity also checks negation)
    assert result["n_scored"] == 2  # _NO_MENTION abstains
    assert abs(result["coverage"] - 2 / 3) < 1e-9
    assert result["accuracy"] == 1.0


def test_audit_lf_false_positive_and_false_negative():
    # Deliberately wrong gold labels to exercise the fp/fn branches.
    result_fp = audit_lf("Medial OA", "keyword_proximity", [_MEDIAL_POSITIVE], [0])
    assert result_fp["fp"] == 1
    result_fn = audit_lf("Medial OA", "negation", [_MEDIAL_NEGATIVE], [1])
    assert result_fn["fn"] == 1


def test_audit_all_covers_every_compartment_and_lf():
    reports = [_MEDIAL_POSITIVE]
    gold_by_compartment = {"Medial OA": [1], "Lateral OA": [0], "PF OA": [0]}
    results = audit_all(reports, gold_by_compartment, translated_reports=reports)
    keys = {(r["compartment"], r["lf_name"]) for r in results}
    assert keys == {
        ("Medial OA", "keyword_proximity"), ("Medial OA", "negation"),
        ("Medial OA", "translate_keyword_proximity"), ("Medial OA", "translate_negation"),
        ("Lateral OA", "keyword_proximity"), ("Lateral OA", "negation"),
        ("Lateral OA", "translate_keyword_proximity"), ("Lateral OA", "translate_negation"),
        ("PF OA", "keyword_proximity"), ("PF OA", "negation"),
        ("PF OA", "translate_keyword_proximity"), ("PF OA", "translate_negation"),
    }


def test_audit_lf_translate_prefix_uses_translated_reports():
    # translate_keyword_proximity should ignore `reports` and use
    # `translated_reports` instead.
    result = audit_lf(
        "Medial OA", "translate_keyword_proximity",
        reports=[_NO_MENTION], gold_values=[1],
        translated_reports=[_MEDIAL_POSITIVE],
    )
    assert result["tp"] == 1


def test_audit_lf_translate_prefix_without_translated_reports_raises():
    import pytest
    with pytest.raises(ValueError):
        audit_lf("Medial OA", "translate_keyword_proximity", [_MEDIAL_POSITIVE], [1])


def test_audit_empty_reports():
    result = audit_lf("Medial OA", "keyword_proximity", [], [])
    assert result["n_total"] == 0
    assert result["coverage"] is None
    assert result["accuracy"] is None
