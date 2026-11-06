"""Pool out-of-fold (OOF) predictions from the DINOv2 fork's fold runs and score them.

Each `train`-mode fold run writes package/oof_f<k>.csv: predictions for the studies
that fold held out, from a model that never trained on them. Pooled over all 5 folds,
every study -- including every one of the 58 gold studies -- has exactly one honest
prediction. (Within a single run, the recipe trains on the gold studies outside its
holdout, so only ~11 gold studies per fold are honest; pooling fixes that.)

Gold scoring goes through data.splits.evaluate_on_gold (the one sanctioned, call-counted
path to the gold set, with a bootstrap CI).

Usage: python scripts/evaluate_oof.py --label "z4 5-fold" <dir-or-csv> [<dir-or-csv> ...]
  (directories are searched recursively for oof_f*.csv)
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from data.splits import evaluate_on_gold  # noqa: E402
from training.metrics import per_column_auc  # noqa: E402
from utils.constants import LABEL_COLUMNS  # noqa: E402


def load_oof(paths):
    files = []
    for p in map(Path, paths):
        files += sorted(p.rglob("oof_f*.csv")) if p.is_dir() else [p]
    if not files:
        raise SystemExit("no oof_f*.csv found")
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    dup = df["StudyInstanceUID"].duplicated()
    if dup.any():
        raise SystemExit(f"{dup.sum()} studies appear in more than one fold's OOF -- "
                         f"mixed runs or overlapping folds? files: {[str(f) for f in files]}")
    return df, files


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="+")
    ap.add_argument("--label", default="", help="run name for the evaluate_on_gold log line")
    args = ap.parse_args()

    oof, files = load_oof(args.paths)
    print(f"{len(files)} OOF file(s), folds {sorted(oof['fold'].unique().tolist())}, "
          f"{len(oof)} studies")

    train = pd.read_csv(ROOT / "data" / "raw" / "train.csv")
    gold = train[train[LABEL_COLUMNS].notna().all(axis=1)].set_index("StudyInstanceUID")
    m = oof.set_index("StudyInstanceUID").join(gold[LABEL_COLUMNS], how="inner", rsuffix="_gold")
    print(f"gold studies with an OOF prediction: {len(m)} of {len(gold)}"
          + ("" if len(m) == len(gold) else "  (partial -- not all folds present)"))
    if len(m) == 0:
        return
    y_true = m[[c + "_gold" for c in LABEL_COLUMNS]].to_numpy().astype(int)
    y_pred = m[LABEL_COLUMNS].to_numpy(dtype=float)
    res = evaluate_on_gold(y_true, y_pred, label=args.label or "dinov2 OOF")
    print(f"OOF gold macro AUC {res['point_estimate']:.4f}  "
          f"95% CI [{res['ci_low']:.4f}, {res['ci_high']:.4f}]  (n={len(m)})")
    per_col = per_column_auc(y_true, y_pred)
    for c in LABEL_COLUMNS:
        print(f"  {c:<17} {per_col[c]:.3f}  "
              f"(gold positives {int(y_true[:, LABEL_COLUMNS.index(c)].sum())})")


if __name__ == "__main__":
    main()
