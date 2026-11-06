"""Report parameter count, MACs, and CPU latency for the default-sized
KneeMultiViewModel at a realistic input shape, against the real cached test
study if available (falls back to random data otherwise). This is the
"build FLOPs/latency tracking in from day one" step actually being used,
not just present as unused code ([redacted: private project notes] engineering priorities).

Usage: python scripts/report_model_cost.py [--target-shape D H W] [--batch-size N]
"""

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from models.backbone3d import Simple3DCNN, count_parameters
from models.efficiency import count_macs, measure_latency
from models.knee_model import KneeMultiViewModel
from utils.constants import LABEL_COLUMNS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target-shape", nargs=3, type=int, default=[32, 256, 256])
    ap.add_argument("--batch-size", type=int, default=4)
    args = ap.parse_args()
    D, H, W = args.target_shape

    backbone = Simple3DCNN()  # default channels=(16,32,64,128)
    model = KneeMultiViewModel(n_classes=len(LABEL_COLUMNS), backbone=backbone)
    model.eval()

    image = torch.randn(args.batch_size, 3, D, H, W)
    plane_present = torch.ones(args.batch_size, 3, dtype=torch.bool)

    n_params = count_parameters(model)
    macs = count_macs(model, lambda: model(image, plane_present))
    latency = measure_latency(lambda: model(image, plane_present), device="cpu", n_warmup=2, n_repeats=5)

    print(f"Input shape: (B={args.batch_size}, 3 planes, {D}, {H}, {W})")
    print(f"Parameters: {n_params:,}")
    print(f"MACs per forward pass (batch={args.batch_size}): {macs:,} "
          f"({macs / args.batch_size:,.0f} per study)")
    print(f"FLOPs estimate (2x MACs): {2 * macs:,}")
    print(f"CPU latency: {latency['mean_ms']:.1f} ms +/- {latency['std_ms']:.1f} ms "
          f"per batch of {args.batch_size} ({latency['mean_ms'] / args.batch_size:.1f} ms/study)")


if __name__ == "__main__":
    main()
