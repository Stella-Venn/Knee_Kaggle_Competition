"""Step 3 of the OA weak-supervision pipeline: audit LF quality against the
gold dev subset. Never touches the locked holdout 18 -- see
data/splits/oa_investigation_split.json and [redacted: private project notes]'s holdout
discipline note (2026-09-04).

Usage: python scripts/audit_oa_lfs.py
"""

import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from data.label_mining.weak_supervision.audit import audit_all
from translate_reports import load_or_build_translations


def main():
    with open(ROOT / "data" / "raw" / "train.csv", newline="", encoding="utf-8") as f:
        rows = list(csv.reader(f))
    header, rows = rows[0], rows[1:]
    label_cols = header[2:]

    with open(ROOT / "data" / "splits" / "oa_investigation_split.json") as f:
        split = json.load(f)
    dev_ids = set(split["dev"])

    dev_rows = [row for row in rows if row[0] in dev_ids and row[2] != ""]
    print(f"n dev rows with gold labels: {len(dev_rows)}")

    reports = [row[1] for row in dev_rows]
    dev_study_ids = [row[0] for row in dev_rows]
    translated_reports = load_or_build_translations(dev_study_ids, reports)

    oa_cols = ["Medial OA", "Lateral OA", "PF OA"]
    gold_by_compartment = {
        col: [int(row[2 + label_cols.index(col)]) for row in dev_rows]
        for col in oa_cols
    }

    results = audit_all(reports, gold_by_compartment, translated_reports)

    print(f"\n{'Compartment':<12} {'LF':<18} {'coverage':>9} {'accuracy':>9} "
          f"{'tp':>4} {'tn':>4} {'fp':>4} {'fn':>4}  verdict")
    for r in results:
        acc_str = f"{r['accuracy']:.1%}" if r["accuracy"] is not None else "n/a"
        cov_str = f"{r['coverage']:.1%}" if r["coverage"] is not None else "n/a"
        if r["accuracy"] is None:
            verdict = "no data"
        elif r["accuracy"] < 0.5:
            verdict = "KILL (below chance)"
        elif r["accuracy"] < 0.65:
            verdict = "WEAK -- down-weight"
        else:
            verdict = "keep"
        print(
            f"{r['compartment']:<12} {r['lf_name']:<18} {cov_str:>9} {acc_str:>9} "
            f"{r['tp']:>4} {r['tn']:>4} {r['fp']:>4} {r['fn']:>4}  {verdict}"
        )


if __name__ == "__main__":
    main()
