"""Unit tests for src/data/label_mining.py.

All report text below is SYNTHETIC, written to reproduce the taxonomy subtleties
this module exists to handle (osteochondral vs. subchondral fracture, "popliteus"
vs. "[redacted: report text]"). Real report text is part of a restricted dataset and is
never committed.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from data.label_mining import (
    mine_bakers,
    mine_contusion,
    mine_effusion,
    mine_fracture,
    mine_report,
    mining_agreement,
)


def test_fracture_negated_mention_is_negative():
    text = "Knee alignment looks normal. No fracture is seen."
    assert mine_fracture(text).value == 0


def test_fracture_osteochondral_is_excluded_english():
    text = (
        "Osteochondral fracture of the lateral patellar facet, 10 mm across, "
        "with a loose OCD fragment."
    )
    assert mine_fracture(text).value == 0


def test_fracture_osteochondral_is_excluded_bulgarian():
    text = "Вижда се остеохондрална фрактура на медиалния кондил."
    assert mine_fracture(text).value == 0


def test_fracture_subchondral_is_positive_turkish():
    # "kırığı" is the inflected form of "kırık" (Turkish root consonant
    # softens k->ğ before a vowel suffix) -- must still match.
    text = "Lateral kondilde subkondral kırığı ve ödem görüldü."
    result = mine_fracture(text)
    assert result.value == 1
    assert result.evidence is not None


def test_fracture_subchondral_insufficiency_is_positive_english():
    text = (
        "Subchondral insufficiency fracture in the medial part of the "
        "tibial plateau, no collapse of the surface."
    )
    assert mine_fracture(text).value == 1


def test_fracture_subarticular_is_positive_turkish_diacritic():
    # "fraktür" (with u-umlaut) must match despite the diacritic.
    text = "Lateral kondilde subartiküler küçük fraktür izlendi."
    assert mine_fracture(text).value == 1


def test_bakers_popliteal_cyst_is_positive():
    text = "A popliteal cyst of 20 x 15 x 30 mm is seen."
    assert mine_bakers(text).value == 1


def test_bakers_popliteus_muscle_is_not_a_false_positive():
    # "popliteus" is a muscle/tendon, unrelated to a Baker's/popliteal cyst.
    # A bare "poplite" stem match would incorrectly fire here.
    text = "The popliteus and biceps tendons look fine."
    assert mine_bakers(text).value == 0


def test_effusion_mild_severity_is_uncertain_not_a_guess():
    # A systematic read of all 58 gold rows found mild/minimal/trace effusion
    # mentions split roughly 50/50 on the gold label -- including two reports
    # from what looks like the same template with near-identical wording and
    # opposite labels. Guessing here was right about half the time; this
    # exact case ("[redacted: report text]") happens to be gold=1, but the
    # miner correctly declines to guess rather than get credit by luck.
    text = "Leve derrame articular. Quistes poplíteos ausentes."
    assert mine_effusion(text).value is None


def test_effusion_strong_severity_is_positive():
    text = "Summary: massive joint effusion and bursitis."
    assert mine_effusion(text).value == 1


def test_effusion_unqualified_mention_is_positive():
    # No severity qualifier ("mild"/"large"/etc.) directly precedes the
    # match -- treated as confident positive, per the "unqualified" bucket
    # (15/20 correct in the systematic check, the best of the three buckets
    # apart from strong-severity).
    text = (
        "The PCL looks fine and its signal is unremarkable throughout the "
        "study. Joint effusion and bursitis with several plicae; "
        "thickened synovium."
    )
    assert mine_effusion(text).value == 1


def test_effusion_negated_is_negative():
    text = "No evidence of knee effusion. Both menisci look fine."
    assert mine_effusion(text).value == 0


def test_contusion_soft_tissue_mention_is_positive():
    text = "Popliteus muscle contusion is seen. No fracture or bone contusion."
    assert mine_contusion(text).value == 1


def test_contusion_negated_is_negative():
    text = "There is no bone contusion. Alignment is fine."
    assert mine_contusion(text).value == 0


def test_contusion_no_mention_is_negative():
    text = "ACL is intact. No fracture is seen."
    assert mine_contusion(text).value == 0


def test_mine_report_returns_none_for_unimplemented_labels():
    result = mine_report("ACL is intact. No fracture is seen.")
    assert result["ACL"].value == 0  # implemented since 2026-09-21
    assert result["Medial OA"].value is None
    assert result["Lateral OA"].value is None
    assert result["PF OA"].value is None
    assert result["Fracture"].value == 0  # implemented, and correctly negative here


def test_mining_agreement_scores_only_implemented_labels():
    reports = [
        "No fracture is seen.",
        "A popliteal cyst of 20mm is seen. Joint effusion present.",
    ]
    gold = [
        {"Fracture": 0, "ACL": 0},
        {"Fracture": 0, "Baker's": 1, "Effusion": 1, "ACL": 1},
    ]

    stats = mining_agreement(reports, gold, columns=["Fracture", "Baker's", "Effusion", "ACL"])

    assert stats["Fracture"]["n_scored"] == 2
    assert stats["Fracture"]["tn"] == 2
    assert stats["Fracture"]["accuracy"] == 1.0

    assert stats["Baker's"]["n_scored"] == 1
    assert stats["Baker's"]["tp"] == 1

    assert stats["Effusion"]["n_scored"] == 1
    assert stats["Effusion"]["tp"] == 1

    # ACL has no miner -> never scored, regardless of gold data present.
    assert stats["ACL"]["n_scored"] == 0
    assert stats["ACL"]["accuracy"] is None


# --- 2026-09-24 pre-retrain fixes -------------------------------------------

from data.label_mining.rule_based import mine_bakers, mine_contusion, mine_fracture  # noqa: E402
from data.text_preprocess import lower_report  # noqa: E402


def test_postposed_none_negates_term_miners():
    # [redacted: study ID]: "[redacted: report text]" was mined POSITIVE.
    assert mine_bakers("Joint effusion: Large.\nBaker cyst: None.\n").value == 0
    # stem "fractur" must check negation from the end of the word ("fracture: none")
    assert mine_fracture("Fracture: None.").value == 0
    assert mine_fracture("Fracture of the tibial plateau, no displacement.").value == 1


def test_bakers_vocabulary_gaps_from_gold_dev():
    assert mine_bakers("Popliteale cyste, zonder complicatie.").value == 1
    assert mine_bakers("Baker cista veličine 2 cm.").value == 1
    assert mine_bakers("Бекерова киста зад коляното.").value == 1


def test_bakers_small_or_trace_abstains_same_sentence_only():
    assert mine_bakers("Trace Baker's cyst.").value is None
    assert mine_bakers("Pequeño quiste poplíteo, sin rotura.").value is None
    # small EFFUSION in the previous sentence must not make a real cyst abstain
    assert mine_bakers("Manji izljev u koljenu.\nBaker cista veličine 2 cm.").value == 1
    # "mala" inside "hondromalacija"[redacted: report text]"small"
    assert mine_bakers("hondromalacija, baker cista 3 cm.").value == 1


def test_contusion_stem_catches_dutch_contusie():
    assert mine_contusion("Vermoedelijk botcontusie, geen fractuur.").value == 1


def test_micro_sign_folded_to_greek_mu():
    # 305/321 Greek reports write mu as U+00B5 in some words.
    assert lower_report("οστεοµυελικό") == "οστεομυελικό"
    assert mine_contusion("Παρατηρείται µώλωπας του οστού.").value == 1
