"""Compare fold runs -- and their rank-mean ensemble -- on the report-label holdout.

Each argument is one recipe: a directory (searched recursively for oof_f*.csv) or an OOF
CSV. Scores use the same target as the fork's in-run "holdout" number: the gold label where
a study has one, else the public LLM label thresholded at 0.5 (upstream's train cell). Only
studies present in EVERY recipe are scored, so the numbers are directly comparable, and the
ensemble is the per-column rank mean of the recipes (as upstream's inference combines members).

This deliberately does NOT go through evaluate_on_gold: it is the cheap, repeatable check for
"does adding recipe B to recipe A help?" (idea 4), not a gold evaluation.

Usage: python scripts/compare_oof.py --llm <report_labels_v2.csv> A=<dir-or-csv> B=<dir-or-csv> ...
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from utils.constants import LABEL_COLUMNS  # noqa: E402


def load(spec):
    p = Path(spec)
    files = sorted(p.rglob("oof_f*.csv")) if p.is_dir() else [p]
    if not files:
        raise SystemExit(f"no oof_f*.csv under {spec}")
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    if df["StudyInstanceUID"].duplicated().any():
        raise SystemExit(f"{spec}: a study appears in more than one OOF file")
    return df.set_index("StudyInstanceUID")[LABEL_COLUMNS]


def macro(y, p):
    a = [roc_auc_score(y[:, j], p[:, j]) if len(set(y[:, j])) > 1 else np.nan
         for j in range(y.shape[1])]
    return float(np.nanmean(a)), a


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("recipes", nargs="+", help="NAME=<dir-or-csv>")
    ap.add_argument("--llm", required=True, help="public LLM label table (report_labels_v2.csv)")
    args = ap.parse_args()
    recs = dict(r.split("=", 1) for r in args.recipes)
    oof = {k: load(v) for k, v in recs.items()}
    common = sorted(set.intersection(*(set(d.index) for d in oof.values())))

    train = pd.read_csv(ROOT / "data" / "raw" / "train.csv").set_index("StudyInstanceUID")
    gold = train[train[LABEL_COLUMNS].notna().all(axis=1)][LABEL_COLUMNS]
    llm = pd.read_csv(args.llm).set_index("StudyInstanceUID")[LABEL_COLUMNS]
    ids = [s for s in common if s in gold.index or s in llm.index]
    y = np.stack([gold.loc[s].values if s in gold.index else (llm.loc[s].values > 0.5)
                  for s in ids]).astype(int)
    print(f"{len(ids)} studies scored (in every recipe; {len(common) - len(ids)} without a target)")

    preds = {k: d.loc[ids].to_numpy(float) for k, d in oof.items()}
    ranks = {k: pd.DataFrame(v).rank(pct=True).to_numpy() for k, v in preds.items()}
    rows = {k: macro(y, v) for k, v in preds.items()}
    if len(preds) > 1:
        rows["rank mean (all)"] = macro(y, np.mean(list(ranks.values()), axis=0))
    w = max(len(k) for k in rows)
    print(f"{'recipe':<{w}}  macro   " + " ".join(f"{c[:8]:>8}" for c in LABEL_COLUMNS))
    for k, (m, a) in rows.items():
        print(f"{k:<{w}}  {m:.4f}  " + " ".join(f"{x:8.3f}" for x in a))


if __name__ == "__main__":
    main()
