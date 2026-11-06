"""Audit the ACL / MCL / Medial Meniscus / Lateral Meniscus / Synovitis rule miners
against the gold labels.

Usage: python scripts/audit_structure_miners.py [--show-errors] [--holdout]

Default scores only the 40-study dev subset (data/splits/oa_investigation_split.json),
which is what miner rules should be tuned against. --holdout scores the 18 held-out
studies instead: run it once per finished rule set, not while iterating. (Caveat: the
split isn't truly blind -- see [redacted: private project notes].) Also prints each miner's coverage and
predicted-positive rate over all 4407 training reports, since abstention rate matters
as much as accuracy on the gold subset.
"""

import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from data.label_mining.rule_based import mine_report
from utils.constants import LABEL_COLUMNS

TARGETS = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Synovitis"]


def main():
    show_errors = "--show-errors" in sys.argv
    use_holdout = "--holdout" in sys.argv
    with open(ROOT / "data" / "raw" / "train.csv", newline="", encoding="utf-8") as f:
        rows = list(csv.reader(f))
    header, rows = rows[0], rows[1:]
    with open(ROOT / "data" / "splits" / "oa_investigation_split.json") as f:
        split = json.load(f)
    wanted = set(split["holdout"] if use_holdout else split["dev"])
    gold_rows = [r for r in rows if r[0] in wanted and any(v != "" for v in r[2:])]
    print(f"Scoring {'HOLDOUT' if use_holdout else 'dev'} subset: {len(gold_rows)} gold studies\n")

    mined_gold = [mine_report(r[1]) for r in gold_rows]
    print(f"{'label':<17}{'scored':>7}{'abstain':>8}{'tp':>4}{'tn':>4}{'fp':>4}{'fn':>4}"
          f"{'acc':>7}{'prec':>7}{'recall':>8}   all-4407: coverage / pos-rate")
    all_mined = [mine_report(r[1]) for r in rows]
    for col in TARGETS:
        j = header.index(col)
        tp = tn = fp = fn = abst = 0
        errors = []
        for r, m in zip(gold_rows, mined_gold):
            pred, truth = m[col].value, int(r[j])
            if pred is None:
                abst += 1
            elif pred == truth:
                tp, tn = tp + (truth == 1), tn + (truth == 0)
            else:
                fp, fn = fp + (pred == 1), fn + (pred == 0)
                errors.append((r[0][-8:], pred, truth, m[col].evidence))
        n = tp + tn + fp + fn
        acc = f"{(tp + tn) / n:.0%}" if n else "n/a"
        prec = f"{tp / (tp + fp):.0%}" if tp + fp else "n/a"
        rec = f"{tp / (tp + fn):.0%}" if tp + fn else "n/a"
        vals = [m[col].value for m in all_mined]
        labeled = [v for v in vals if v is not None]
        cov = len(labeled) / len(vals)
        posr = sum(labeled) / len(labeled) if labeled else float("nan")
        print(f"{col:<17}{n:>7}{abst:>8}{tp:>4}{tn:>4}{fp:>4}{fn:>4}{acc:>7}{prec:>7}{rec:>8}"
              f"   {cov:.0%} / {posr:.0%}")
        if show_errors:
            for sid, pred, truth, ev in errors:
                print(f"    {sid} mined={pred} gold={truth}  {ev!r}")


if __name__ == "__main__":
    main()
