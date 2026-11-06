"""Tests for src/data/text_preprocess.py.

These utilities moved here from label_mining/rule_based.py (module 5, 2026-09-22);
tests/test_structure_miners.py and tests/test_label_mining.py already cover them
indirectly through the miners, so this file focuses on the module's own contract
(including the parts no miner happens to exercise) rather than re-deriving every
case those files already check.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from data.text_preprocess import (
    detect_language,
    is_negated,
    is_negated_forward,
    is_negated_whole_word,
    split_clauses,
)


# --- split_clauses ------------------------------------------------------------

def test_split_clauses_basic_sentences():
    assert split_clauses("No fracture. ACL is intact.") == ["no fracture", "acl is intact"]


def test_split_clauses_does_not_split_decimal_numbers():
    assert split_clauses("Cyst measuring 3.2 cm in size.") == ["cyst measuring 3.2 cm in size"]


def test_split_clauses_joins_wrapped_lines():
    text = "Medial meniscus has\nno evidence of tear."
    assert split_clauses(text) == ["medial meniscus has no evidence of tear"]


def test_split_clauses_splits_on_bullet_lines():
    text = "Findings:\n- ACL intact\n- MCL intact"
    assert split_clauses(text) == ["findings:", "- acl intact", "- mcl intact"]


def test_split_clauses_splits_missing_space_after_period():
    assert split_clauses("görüldü.Lateral menisküste") == ["görüldü", "lateral menisküste"]


def test_split_clauses_normalizes_micro_sign_to_greek_mu():
    # U+00B5 (micro sign) vs U+03BC (Greek mu) look identical but are different code points;
    # reports mix them, so both must normalize to the same clause text.
    assert split_clauses("έξω µηνίσκος") == split_clauses("έξω μηνίσκος")


def test_split_clauses_drops_empty_clauses():
    assert split_clauses("ACL intact..\n\nMCL intact.") == ["acl intact", "mcl intact"]


# --- is_negated (substring, sentence-bounded) ----------------------------------

def test_is_negated_true_within_same_sentence():
    text = "no evidence of fracture is seen".lower()
    assert is_negated(text, text.index("fracture")) is True


def test_is_negated_does_not_cross_a_period():
    text = "there is no subchondral edema. osteophytes noted".lower()
    assert is_negated(text, text.index("osteophytes")) is False


# --- is_negated_forward (post-posed, template convention) ---------------------

def test_is_negated_forward_true_after_colon():
    text = "degenerative change:\nnone.".lower()
    assert is_negated_forward(text, text.index(":")) is True


def test_is_negated_forward_false_across_a_period():
    text = "chondrosis. no new fracture".lower()
    assert is_negated_forward(text, text.index("chondrosis") + len("chondrosis")) is False


# --- is_negated_whole_word (clause-scoped, whole-word) -------------------------

def test_is_negated_whole_word_does_not_false_positive_on_substring():
    # "interno" contains "no " as a substring -- is_negated_whole_word must not
    # be fooled by that the way a naive substring check would be.
    clause = "rotura en menisco interno"
    start = clause.index("rotura")
    assert is_negated_whole_word(clause, start, start + len("rotura")) is False


def test_is_negated_whole_word_true_for_whole_word_match():
    clause = "no hay rotura de menisco"
    start = clause.index("rotura")
    assert is_negated_whole_word(clause, start, start + len("rotura")) is True


def test_is_negated_whole_word_stops_at_comma():
    # "no" appears before the comma but belongs to a different clause-internal phrase
    clause = "no signal changes noted, rotura de menisco medial"
    start = clause.index("rotura")
    assert is_negated_whole_word(clause, start, start + len("rotura")) is False


def test_is_negated_whole_word_forward_tail():
    clause = "yırtık izlenmemiştir"
    end = clause.index(" izlenmemiştir")
    assert is_negated_whole_word(clause, 0, end) is True


# --- detect_language ------------------------------------------------------------

def test_detect_language_identifies_clear_examples():
    assert detect_language("The anterior cruciate ligament looks healthy with no evidence of tear.") == "en"
    assert detect_language("Se observa una rotura completa del ligamento cruzado con retracción proximal.") == "es"


def test_detect_language_falls_back_to_en_on_empty_input():
    assert detect_language("") == "en"


def test_detect_language_is_reproducible():
    # langdetect is otherwise randomized; detect_language seeds it every call
    text = "Le ligament du genou présente une rupture complète."
    assert detect_language(text) == detect_language(text)
