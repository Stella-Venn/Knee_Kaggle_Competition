"""Run a trained checkpoint against the 58 gold studies via evaluate_on_gold
(splits.py) -- the ONLY sanctioned way to check a model against the gold set,
per [redacted: private project notes]'s "evaluate sparingly" rule. Every call is logged with a running
count (see evaluate_on_gold's own docstring); don't loop this over many
candidates -- reserve it for a deliberately-chosen model, per [redacted: private project notes].

Usage: python scripts/evaluate_gold.py path/to/checkpoint.pt --label "some name"
       [--cache-dir data/processed/volume_cache/3x0.5x0.5] [--target-shape 32 256 256]
"""

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from data.dataset import KneeStudyDataset
from data.splits import evaluate_on_gold
from models.knee_model import KneeMultiViewModel
from training.engine import predict
from utils.constants import LABEL_COLUMNS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint")
    ap.add_argument("--label", default="")
    ap.add_argument("--cache-dir", default=str(ROOT / "data" / "processed" / "volume_cache" / "3x0.5x0.5"))
    ap.add_argument("--target-shape", nargs=3, type=int, default=[32, 256, 256])
    ap.add_argument("--manifest", default=str(ROOT / "data" / "processed" / "manifest" / "series_manifest.csv"))
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    with open(ROOT / "data" / "raw" / "train.csv", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    gold_rows = [r for r in rows if all(r[c] != "" for c in LABEL_COLUMNS)]
    print(f"{len(gold_rows)} gold studies found in train.csv")
    gold_ids = [r["StudyInstanceUID"] for r in gold_rows]
    y_true = np.array([[int(r[c]) for c in LABEL_COLUMNS] for r in gold_rows], dtype=np.int8)

    series_by_study = defaultdict(list)
    with open(args.manifest, newline="") as f:
        for row in csv.DictReader(f):
            if row["StudyInstanceUID"] in set(gold_ids):
                series_by_study[row["StudyInstanceUID"]].append(row)

    ds = KneeStudyDataset(gold_ids, y_true, series_by_study, cache_dir=args.cache_dir,
                           target_shape=tuple(args.target_shape))
    loader = torch.utils.data.DataLoader(ds, batch_size=8, shuffle=False)

    model = KneeMultiViewModel(n_classes=len(LABEL_COLUMNS)).to(args.device)
    state = torch.load(args.checkpoint, map_location=args.device)
    model.load_state_dict(state)

    study_ids_out, y_true_out, y_pred, mask = predict(model, loader, args.device)
    assert study_ids_out == gold_ids, "row order mismatch between dataset and gold_ids"
    assert mask.all(), "expected every gold cell to be a real label (confirmed no masked cells, module 8 notes)"

    result = evaluate_on_gold(y_true_out, y_pred, columns=LABEL_COLUMNS, label=args.label or args.checkpoint)
    print(f"\nmacro_auc = {result['point_estimate']:.4f}  "
          f"95% CI [{result['ci_low']:.4f}, {result['ci_high']:.4f}]  "
          f"(n_resamples_used={result['n_resamples_used']}/{result['n_resamples_requested']})")


if __name__ == "__main__":
    main()
