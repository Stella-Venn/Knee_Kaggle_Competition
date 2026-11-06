"""Unit tests for src/data/label_mining/weak_supervision/translate.py.

Does NOT load the real NLLB model (slow, network/disk-dependent) -- tests
translate_batch's batching/routing logic against a fake tokenizer/model,
and checks the language-code mapping table directly. Real translation
quality is checked empirically via scripts/audit_translate_then_match.py
against gold labels, not asserted here.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from data.label_mining.weak_supervision.translate import LANGDETECT_TO_NLLB, translate_batch


class _FakeInputs(dict):
    def to(self, device):
        return self


class _FakeTokenizer:
    """Minimal stand-in: "translates" by uppercasing, to make routing
    (which texts got sent through the fake model) verifiable without a
    real model.
    """

    def __init__(self):
        self.src_lang = None

    def convert_tokens_to_ids(self, token):
        return 0

    def __call__(self, texts, **kwargs):
        return _FakeInputs(texts=texts)

    def batch_decode(self, out, skip_special_tokens=True):
        return out  # `out` is already the "translated" text list


class _FakeModel:
    def generate(self, **kwargs):
        return [t.upper() for t in kwargs["texts"]]


def test_english_passes_through_unchanged():
    tok, model = _FakeTokenizer(), _FakeModel()
    texts = ["ACL is intact."]
    langs = ["eng_Latn"]
    result = translate_batch(texts, langs, tok, model, device="cpu")
    assert result == ["ACL is intact."]  # unchanged, not uppercased


def test_non_english_gets_routed_through_model():
    tok, model = _FakeTokenizer(), _FakeModel()
    texts = ["derrame articular"]
    langs = ["spa_Latn"]
    result = translate_batch(texts, langs, tok, model, device="cpu")
    assert result == ["DERRAME ARTICULAR"]  # went through the fake "model"


def test_mixed_languages_preserve_order():
    tok, model = _FakeTokenizer(), _FakeModel()
    texts = ["hello", "hola", "world", "mundo"]
    langs = ["eng_Latn", "spa_Latn", "eng_Latn", "spa_Latn"]
    result = translate_batch(texts, langs, tok, model, device="cpu")
    assert result == ["hello", "HOLA", "world", "MUNDO"]


def test_langdetect_mapping_covers_all_dataset_languages():
    # From the Phase 1 language-distribution finding ([redacted: private project notes]):
    # en/es/tr/hr/el/de/bg/nl/fr, ~39/16/12/9/7/6/5/4/2%.
    expected_langdetect_codes = {"en", "es", "tr", "hr", "el", "de", "bg", "nl", "fr"}
    assert expected_langdetect_codes == set(LANGDETECT_TO_NLLB.keys())


def test_langdetect_mapping_values_are_valid_flores_codes():
    # Spot-check the format (xxx_Yyyy), not an exhaustive FLORES-200 check.
    import re
    for code in LANGDETECT_TO_NLLB.values():
        assert re.match(r"^[a-z]{3}_[A-Z][a-z]{3}$", code), code
