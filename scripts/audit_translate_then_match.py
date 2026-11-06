"""Empirical check of translate-then-match against the gold dev subset --
does NOT decide translation quality from a couple of spot-checked
sentences, measures it the same way every other LF here has been measured.

Usage: python scripts/audit_translate_then_match.py
Caches translations to data/processed/dev_translations.json (dev subset
only -- full-pool translation is a separate, later step once this is
validated).
"""

import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from data.label_mining.weak_supervision.audit import audit_lf
from data.label_mining.weak_supervision.oa_lfs import COMPARTMENTS, make_keyword_proximity_lf
from data.label_mining.weak_supervision.translate import LANGDETECT_TO_NLLB, load_translator, translate_batch
from data.text_preprocess import detect_language


def main():
    with open(ROOT / "data" / "raw" / "train.csv", newline="", encoding="utf-8") as f:
        rows = list(csv.reader(f))
    header, rows = rows[0], rows[1:]
    label_cols = header[2:]

    with open(ROOT / "data" / "splits" / "oa_investigation_split.json") as f:
        split = json.load(f)
    dev_ids = set(split["dev"])
    dev_rows = [row for row in rows if row[0] in dev_ids and row[2] != ""]
    print(f"n dev rows: {len(dev_rows)}")

    cache_path = ROOT / "data" / "processed" / "dev_translations.json"
    if cache_path.exists():
        with open(cache_path) as f:
            cache = json.load(f)
        print(f"Loaded {len(cache)} cached translations")
    else:
        cache = {}

    to_translate_ids = [row[0] for row in dev_rows if row[0] not in cache]
    if to_translate_ids:
        print(f"Translating {len(to_translate_ids)} uncached reports...")
        id_to_text = {row[0]: row[1] for row in dev_rows}
        texts = [id_to_text[sid] for sid in to_translate_ids]
        langdetect_codes = [detect_language(t) for t in texts]
        nllb_codes = [LANGDETECT_TO_NLLB.get(c, "eng_Latn") for c in langdetect_codes]

        tokenizer, model, device = load_translator()
        translated = translate_batch(texts, nllb_codes, tokenizer, model, device)
        for sid, lang, text in zip(to_translate_ids, langdetect_codes, translated):
            cache[sid] = {"lang": lang, "translated": text}
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_path, "w") as f:
            json.dump(cache, f, ensure_ascii=False, indent=2)
        print(f"Saved {len(cache)} translations to {cache_path}")

    translated_reports = [cache[row[0]]["translated"] for row in dev_rows]

    print(f"\n{'Compartment':<12} {'coverage':>9} {'accuracy':>9} {'tp':>4} {'tn':>4} {'fp':>4} {'fn':>4}")
    for compartment in COMPARTMENTS:
        idx = label_cols.index(compartment)
        gold = [int(row[2 + idx]) for row in dev_rows]
        result = audit_lf(compartment, "keyword_proximity", translated_reports, gold)
        acc_str = f"{result['accuracy']:.1%}" if result["accuracy"] is not None else "n/a"
        cov_str = f"{result['coverage']:.1%}" if result["coverage"] is not None else "n/a"
        print(f"{compartment:<12} {cov_str:>9} {acc_str:>9} "
              f"{result['tp']:>4} {result['tn']:>4} {result['fp']:>4} {result['fn']:>4}")


if __name__ == "__main__":
    main()
