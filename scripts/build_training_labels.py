"""Step 6: threshold the OA soft labels and assemble one training label table
from every source (gold, rule-based miners, weak-supervision OA labels).

Usage: python scripts/build_training_labels.py
Needs data/raw/train.csv and data/processed/oa_soft_labels.npz (run
fit_oa_label_model.py first). Writes data/processed/training_labels.npz.

The gold-agreement check at the end uses only the OA dev subset, per the
holdout discipline in data/splits/oa_investigation_split.json.
"""

import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from data.label_mining.final_labels import (
    MASKED,
    assemble_training_labels,
    label_weights,
    label_summary,
    rule_label_matrix,
    threshold_soft_labels,
    unmentioned_negative_matrix,
)
from data.label_mining.weak_supervision.oa_lfs import COMPARTMENTS
from utils.constants import LABEL_COLUMNS

POS_THRESHOLD = 0.85
NEG_THRESHOLD = 0.15


def main():
    with open(ROOT / "data" / "raw" / "train.csv", newline="", encoding="utf-8") as f:
        rows = list(csv.reader(f))
    header, rows = rows[0], rows[1:]
    label_cols = header[2:]
    assert label_cols == LABEL_COLUMNS, "train.csv label columns differ from LABEL_COLUMNS"
    study_ids = [row[0] for row in rows]
    reports = [row[1] for row in rows]
    n = len(rows)

    gold = np.full((n, len(LABEL_COLUMNS)), MASKED, dtype=np.int8)
    for i, row in enumerate(rows):
        for j, v in enumerate(row[2:]):
            if v != "":
                gold[i, j] = int(v)

    rule = rule_label_matrix(reports)

    soft_path = ROOT / "data" / "processed" / "oa_soft_labels.npz"
    soft = np.load(soft_path, allow_pickle=True)
    if list(soft["study_ids"]) != study_ids:
        raise SystemExit("oa_soft_labels.npz doesn't match train.csv row order; refit first")
    weak = np.full((n, len(LABEL_COLUMNS)), MASKED, dtype=np.int8)
    print(f"OA soft labels thresholded at >{POS_THRESHOLD} / <{NEG_THRESHOLD}:")
    for compartment in COMPARTMENTS:
        s = soft[f"{compartment}_soft"]
        hard = threshold_soft_labels(s, POS_THRESHOLD, NEG_THRESHOLD)
        weak[:, LABEL_COLUMNS.index(compartment)] = hard
        print(f"  {compartment:<11} positive={int((hard == 1).sum()):>5} "
              f"negative={int((hard == 0).sum()):>5} masked={int((hard == MASKED).sum()):>5} "
              f"(of which no signal at all: {int((s == 0.5).sum())})")

    unmentioned = unmentioned_negative_matrix(reports)
    labels, source = assemble_training_labels(gold, rule, weak, unmentioned)

    out_path = ROOT / "data" / "processed" / "training_labels.npz"
    np.savez(
        out_path,
        study_ids=np.array(study_ids, dtype=object),
        labels=labels,
        source=source,
        weights=label_weights(source),
        columns=np.array(LABEL_COLUMNS, dtype=object),
    )
    print(f"\nSaved {labels.shape} label table to {out_path}")

    print(f"\n{'Label':<18} {'pos':>5} {'neg':>5} {'masked':>7}  sources")
    for r in label_summary(labels, source):
        print(f"{r['label']:<18} {r['n_positive']:>5} {r['n_negative']:>5} "
              f"{r['n_masked']:>7}  {r['sources']}")

    with open(ROOT / "data" / "splits" / "oa_investigation_split.json") as f:
        dev_ids = set(json.load(f)["dev"])
    dev_idx = [i for i, sid in enumerate(study_ids) if sid in dev_ids and (gold[i] != MASKED).any()]
    print(f"\nWeak-supervision OA labels vs. gold on the {len(dev_idx)}-study dev subset "
          f"(agreement among studies given a confident label):")
    for compartment in COMPARTMENTS:
        j = LABEL_COLUMNS.index(compartment)
        confident = [i for i in dev_idx if weak[i, j] != MASKED]
        agree = sum(weak[i, j] == gold[i, j] for i in confident)
        acc = f"{agree / len(confident):.1%}" if confident else "n/a"
        print(f"  {compartment:<11} confident on {len(confident):>2}/{len(dev_idx)}, agreement {acc}")


if __name__ == "__main__":
    main()
