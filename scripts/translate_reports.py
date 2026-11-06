"""Translate all training reports to English, cached to
data/processed/report_translations.json (keyed by StudyInstanceUID).
Shared by build_oa_vote_matrix.py and audit_oa_lfs.py so translation only
ever runs once, not per-script.

Usage: python scripts/translate_reports.py
"""

import csv
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

CACHE_PATH = ROOT / "data" / "processed" / "report_translations.json"


def _save_cache(cache):
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = CACHE_PATH.with_suffix(".json.tmp")
    with open(tmp, "w") as f:
        json.dump(cache, f, ensure_ascii=False, indent=2)
    tmp.replace(CACHE_PATH)  # atomic-ish swap, avoids a half-written cache file


def load_or_build_translations(study_ids, reports, device=None, chunk_size=200):
    """Returns list[str] of English text, same order as `reports` -- from
    cache where available, translating (and caching) anything new.

    Checkpoints the cache to disk after every `chunk_size` reports, not
    just once at the end -- a run that needs to be killed partway through
    (e.g. to diagnose slowness) doesn't lose all its progress. device:
    forwarded to translate.load_translator (None = auto-detect).
    """
    if CACHE_PATH.exists():
        with open(CACHE_PATH) as f:
            cache = json.load(f)
    else:
        cache = {}

    missing = [(sid, text) for sid, text in zip(study_ids, reports) if sid not in cache]
    if not missing:
        print(f"All {len(study_ids)} translations already cached")
        return [cache[sid]["translated"] for sid in study_ids]

    from data.label_mining.weak_supervision.translate import (
        LANGDETECT_TO_NLLB,
        load_translator,
        translate_batch,
    )
    from data.text_preprocess import detect_language

    print(f"Translating {len(missing)} uncached reports...")
    tokenizer, model, device = load_translator(device=device)
    print(f"Using device: {device}")

    start_time = time.time()
    n_translated_total = 0
    for chunk_start in range(0, len(missing), chunk_size):
        chunk = missing[chunk_start:chunk_start + chunk_size]
        chunk_ids = [sid for sid, _ in chunk]
        chunk_texts = [text for _, text in chunk]

        langdetect_codes = [detect_language(t) for t in chunk_texts]
        nllb_codes = [LANGDETECT_TO_NLLB.get(c, "eng_Latn") for c in langdetect_codes]

        def _progress(n_done, n_total, chunk_start=chunk_start):
            elapsed = time.time() - start_time
            done_total = chunk_start + n_done
            rate = done_total / elapsed if elapsed > 0 else 0
            eta = (len(missing) - done_total) / rate if rate > 0 else float("nan")
            print(
                f"  chunk batch: {n_done}/{n_total} in this chunk | "
                f"{done_total}/{len(missing)} overall | "
                f"{elapsed:.0f}s elapsed | ~{rate:.2f} reports/s | ETA {eta/60:.1f} min",
                flush=True,
            )

        translated = translate_batch(chunk_texts, nllb_codes, tokenizer, model, device, on_batch_done=_progress)
        for sid, lang, text in zip(chunk_ids, langdetect_codes, translated):
            cache[sid] = {"lang": lang, "translated": text}
        _save_cache(cache)
        n_translated_total += len(chunk)
        print(f"Checkpointed {n_translated_total}/{len(missing)} "
              f"({len(cache)} total cached) to {CACHE_PATH}", flush=True)

    print(f"Done in {time.time()-start_time:.0f}s")
    return [cache[sid]["translated"] for sid in study_ids]


def main():
    with open(ROOT / "data" / "raw" / "train.csv", newline="", encoding="utf-8") as f:
        rows = list(csv.reader(f))
    _, rows = rows[0], rows[1:]
    study_ids = [row[0] for row in rows]
    reports = [row[1] for row in rows]
    # CPU was tested as a diagnostic and measured at ~0.13 reports/s (~558
    # min ETA for the full set) -- dramatically worse than MPS, not better.
    # The earlier MPS run's low %CPU reading was normal GPU-offloaded
    # behavior, not a sign of being stuck -- that hypothesis was wrong.
    # Back to auto-detect (MPS), now with real checkpointing/progress
    # visibility this time so we get a trustworthy throughput number.
    load_or_build_translations(study_ids, reports, device=None)


if __name__ == "__main__":
    main()
