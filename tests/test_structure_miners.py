"""Tests for the ACL / MCL / meniscus / Synovitis miners in rule_based.py.

All report text below is SYNTHETIC, written to reproduce real template and language
patterns. Real report text is part of a restricted dataset and is never committed.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from data.label_mining.rule_based import (
    mine_acl,
    mine_lateral_meniscus,
    mine_mcl,
    mine_medial_meniscus,
    mine_report,
    mine_synovitis,
)
from data.text_preprocess import split_clauses as _split_clauses


def values(text):
    return (
        mine_acl(text).value, mine_mcl(text).value,
        mine_medial_meniscus(text).value, mine_lateral_meniscus(text).value,
        mine_synovitis(text).value,
    )


# --- clause splitting -------------------------------------------------------

def test_split_joins_wrapped_lines_and_headers():
    text = "Menisco medial de\nforma conservada, con señal aumentada.\nMedial meniscus:\nNormal."
    assert _split_clauses(text) == [
        "menisco medial de forma conservada, con señal aumentada",
        "medial meniscus: normal",
    ]


def test_split_handles_missing_space_and_decimals():
    assert _split_clauses("görüldü.Lateral menisküste") == ["görüldü", "lateral menisküste"]
    assert _split_clauses("baker's cyst 1.5*1.5 cm") == ["baker's cyst 1.5*1.5 cm"]


# --- ligaments --------------------------------------------------------------

def test_acl_normal_template():
    text = "Anterior cruciate ligament (ACL): Normal, fibers intact.\nPosterior cruciate ligament (PCL): Fine."
    assert mine_acl(text).value == 0


def test_acl_rupture_positive_and_mcl_from_normal_list():
    text = "ACL is ruptured at mid body.\nNormal PCL, MCL and LCL."
    assert mine_acl(text).value == 1
    assert mine_mcl(text).value == 0


def test_acl_interstitial_tear_abstains():
    text = "In the notch, the anterior cruciate ligament shows an interstitial tear."
    assert mine_acl(text).value is None


def test_acl_not_mentioned_abstains():
    assert mine_acl("Large effusion.").value is None


def test_pcl_normal_does_not_count_for_acl():
    assert mine_acl("Posterior cruciate ligament (PCL): Fine.").value is None


def test_mcl_grade_ii_positive_but_lcl_grade_ii_is_not_mcl():
    lcl = ("The medial collateral ligament looks intact.\n"
           "Partial tear (grade II sprain) in the lateral collateral ligament.")
    assert mine_mcl(lcl).value == 0
    mcl = "Partial tear (grade II sprain) in the medial collateral ligament."
    assert mine_mcl(mcl).value == 1


def test_general_plural_ligament_statement_sets_both_normal():
    text = "Ligamentos cruzados y colaterales de aspecto normales."
    assert mine_acl(text).value == 0 and mine_mcl(text).value == 0


def test_general_statement_cannot_create_a_positive():
    # tear belongs to the ACL clause; the trailing general mention must not make MCL positive
    text = "Rotura completa del LCA en relación a\nLCP y ligamentos colaterales sin cambios."
    assert mine_acl(text).value == 1
    assert mine_mcl(text).value != 1


def test_french_and_croatian_ligaments():
    assert mine_acl("Ligament croisé antérieur : Normal.\nLigament postérieur : Normal.").value == 0
    assert mine_acl("Prednji križni ligament je urednog signala i kontinuiteta.").value == 0


# --- meniscus ---------------------------------------------------------------

def test_medial_tear_lateral_untouched_and_interno_is_not_negation():
    text = "Rotura en menisco interno."
    assert mine_medial_meniscus(text).value == 1  # "interno" contains "no " -- must not negate
    assert mine_lateral_meniscus(text).value is None


def test_both_menisci_normal():
    assert values("Normal medial and lateral menisci seen.")[2:4] == (0, 0)


def test_templated_medial_normal_lateral_meniscectomy():
    text = ("Medial meniscus: Normal shape and signal.\n"
            "Lateral meniscus: Small body and rear horn "
            "in keeping with a prior partial meniscectomy.")
    assert mine_medial_meniscus(text).value == 0
    assert mine_lateral_meniscus(text).value == 1


def test_compartment_scoped_not_torn():
    text = "In the medial compartment, meniscus not torn and cartilage looks fine."
    assert mine_medial_meniscus(text).value == 0


def test_hedged_tear_and_degeneration_abstain():
    hedged = "R/O ramp lesion or peripheral tear of the rear horn of medial meniscus."
    assert mine_medial_meniscus(hedged).value is None
    assert mine_medial_meniscus("Mucoid degeneration of the medial meniscus.").value is None


def test_wrapped_spanish_meniscus_line_is_normal():
    text = "Menisco medial de\nforma conservada, con señal aumentada en su cuerno posterior."
    assert mine_medial_meniscus(text).value == 0


def test_turkish_no_space_between_sides():
    text = ("Sonuç: Medial menisküs arka hornda radial yırtık görüldü."
            "Lateral menisküste grade II dejenerasyon görüldü.")
    assert mine_medial_meniscus(text).value == 1
    assert mine_lateral_meniscus(text).value is None


def test_german_compound_sides_and_negated_riss():
    text = "Innen- sowie Außenmeniskus unauffällig, ohne Anhalt für einen Riss."
    assert values(text)[2:4] == (0, 0)


def test_bulgarian_and_greek_meniscus():
    assert mine_medial_meniscus("Руптура на задния рог на медиалния менискус.").value == 1
    assert mine_lateral_meniscus("Απλή εκφύλιση στον έξω µηνίσκο.").value is None


# --- synovitis --------------------------------------------------------------

def test_synovitis_explicit_and_mild_and_absent():
    assert mine_synovitis("Effusion and synovitis.").value == 1
    assert mine_synovitis("Mild synovitis of the knee.").value is None
    assert mine_synovitis("Large effusion.").value is None


# --- integration ------------------------------------------------------------

def test_mine_report_covers_the_five_new_labels():
    out = mine_report("ACL is ruptured.\nTear of the medial meniscus.\nEffusion and synovitis.")
    assert out["ACL"].value == 1
    assert out["Medial Meniscus"].value == 1
    assert out["Synovitis"].value == 1
    assert out["Lateral Meniscus"].value is None
