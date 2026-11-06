"""Translation for the translate_then_match LF ([redacted: private project notes], 2026-09-04
decision entry -- offline via HuggingFace transformers, not a cloud API:
reuses infrastructure the project needs anyway for the actual image+text
model, no API costs/credentials/internet dependency for a training-time
step, fully cacheable).

Model: facebook/nllb-200-distilled-600M. One model for all 8 non-English
languages found in the reports (es/tr/hr/el/de/bg/nl/fr), instead of
managing 8 separate language-pair models.

Quality caveat, found and kept, not hidden: spot-checking real dev-subset
sentences found real medical-terminology errors -- "[redacted: report text]"
(joint effusion) mistranslated as "joint rash", and a Turkish sentence
dropped "edema" entirely while getting the rest right. Not assumed good
enough on inspection -- see audit_oa_lfs.py's per-LF accuracy numbers once
this feeds into an LF for the actual, measured answer.

NLLB source-language codes (FLORES-200), matched to what langdetect
returns for the languages actually found in this dataset's reports.
"""

LANGDETECT_TO_NLLB = {
    "en": "eng_Latn",
    "es": "spa_Latn",
    "tr": "tur_Latn",
    "hr": "hrv_Latn",
    "el": "ell_Grek",
    "de": "deu_Latn",
    "bg": "bul_Cyrl",
    "nl": "nld_Latn",
    "fr": "fra_Latn",
}

MODEL_NAME = "facebook/nllb-200-distilled-600M"


def load_translator(model_name: str = MODEL_NAME, device: str | None = None):
    """Returns (tokenizer, model, device). Loads once -- callers should
    reuse this across all translate_batch calls, not reload per batch.

    device: force "cpu"/"mps"/"cuda", or None to auto-detect (prefers MPS
    on Apple Silicon). A run defaulting to MPS took 50+ minutes with no
    clear explanation (no beam search, low %CPU utilization throughout) --
    forcing CPU is a useful diagnostic to rule out an MPS-specific issue,
    not a claim that CPU is generally faster.
    """
    import torch
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    if device is None:
        device = "mps" if torch.backends.mps.is_available() else "cpu"
    model = model.to(device)
    return tokenizer, model, device


def translate_batch(
    texts, src_lang_codes, tokenizer, model, device,
    max_length=512, batch_size=16, on_batch_done=None,
):
    """texts: list[str]. src_lang_codes: list[str], same length, NLLB codes
    (e.g. "spa_Latn") -- use LANGDETECT_TO_NLLB to convert from langdetect
    output. English texts are passed through unchanged, not translated.
    Returns list[str], same order/length as `texts`.

    on_batch_done, if given, is called after every model-generation batch as
    on_batch_done(n_translated_so_far, n_to_translate) -- for progress
    logging/checkpointing on long runs. n_to_translate excludes English
    passthrough texts, since those never reach the model.
    """
    results = [None] * len(texts)
    by_lang = {}
    for i, (text, lang) in enumerate(zip(texts, src_lang_codes)):
        if lang == "eng_Latn":
            results[i] = text
            continue
        by_lang.setdefault(lang, []).append(i)

    n_to_translate = sum(len(idxs) for idxs in by_lang.values())
    n_done = 0
    forced_bos = tokenizer.convert_tokens_to_ids("eng_Latn")
    for lang, idxs in by_lang.items():
        tokenizer.src_lang = lang
        for start in range(0, len(idxs), batch_size):
            batch_idxs = idxs[start:start + batch_size]
            batch_texts = [texts[i] for i in batch_idxs]
            inputs = tokenizer(
                batch_texts, return_tensors="pt", truncation=True,
                max_length=max_length, padding=True,
            ).to(device)
            out = model.generate(**inputs, forced_bos_token_id=forced_bos, max_length=max_length)
            decoded = tokenizer.batch_decode(out, skip_special_tokens=True)
            for i, text in zip(batch_idxs, decoded):
                results[i] = text
            n_done += len(batch_idxs)
            if on_batch_done is not None:
                on_batch_done(n_done, n_to_translate)
    return results
