"""Create data/splits/oa_investigation_split.json: a dev/holdout split of the expert-labeled
studies, stratified by whether a study has any OA label. The original file lists study IDs, so it
is not committed. A new seed gives a different split from the one used in the original work.

Usage: python scripts/make_oa_split.py [--train data/raw/train.csv] [--out <path>] [--seed 0]
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OA = ["Medial OA", "Lateral OA", "PF OA"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default=ROOT / "data" / "raw" / "train.csv")
    ap.add_argument("--out", default=ROOT / "data" / "splits" / "oa_investigation_split.json")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--holdout-fraction", type=float, default=0.3)
    args = ap.parse_args()

    tr = pd.read_csv(args.train)
    gold = tr[tr[OA].notna().all(axis=1)]
    rng = np.random.default_rng(args.seed)
    dev, holdout = [], []
    for _, group in gold.groupby(gold[OA].max(axis=1)):
        ids = group["StudyInstanceUID"].tolist()
        rng.shuffle(ids)
        k = round(len(ids) * args.holdout_fraction)
        holdout += ids[:k]
        dev += ids[k:]
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"dev": sorted(dev), "holdout": sorted(holdout), "seed": args.seed,
                               "holdout_fraction": args.holdout_fraction}, indent=2))
    print(f"{len(dev)} dev, {len(holdout)} holdout -> {out}")


if __name__ == "__main__":
    main()
