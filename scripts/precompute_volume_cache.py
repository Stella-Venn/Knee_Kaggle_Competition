"""Precompute the per-series volume cache the Dataset (module 6) reads from.

This is the "preprocess DICOM to normalized tensors exactly once" step ([redacted: private project notes]
working agreement): resamples every series to one fixed physical spacing and
percentile-normalizes it, writing one .npy per series. `KneeStudyDataset` only
ever reads from this cache -- it never opens a raw .dcm -- so a training run
never touches raw DICOM even indirectly.

The cache is keyed by target spacing (data/processed/volume_cache/<spacing_tag>/),
not just by study/series, so changing target_spacing can't silently serve a stale
cache computed at a different spacing.

**Cache-time crop/pad to a bounded shape (2026-09-22, real Kaggle failure)**: the
first full 558-study pilot run on Kaggle died mid-run (ERROR status, no traceback
recoverable, ~200/558 studies processed, ~10GB downloaded) after resample-only
output turned out to average ~16.6MB/series -- roughly 2x the (32,256,256)-based
estimate this project had been using, because resampling to a fixed SPACING
leaves array size proportional to each study's actual physical field of view,
unbounded. `center_crop_or_pad` (volume_loader.py) is now applied here too, not
just at Dataset __getitem__ time, so cached file size is bounded and predictable
regardless of any one study's real extent. This does mean the cache can only be
cropped SMALLER later, not padded larger, without recomputing -- an accepted
tradeoff, same category as the existing "cache keyed by target_spacing" one.

Usage: python scripts/precompute_volume_cache.py --raw-root data/raw/train_series
       [--manifest data/processed/manifest/series_manifest.csv] [--limit N]
       [--target-spacing 3.0 0.5 0.5] [--target-shape 32 256 256]

This ran locally against the 1 downloaded test study to prove the path end to
end; the full training set's ~1.4TB of raw DICOM isn't (and won't be) stored
locally, so the full-dataset run happens elsewhere -- a Kaggle kernel, following
[redacted: note about institution-specific computing resources]
[redacted: private project notes]).
"""

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from data.dataset import spacing_tag
from data.volume_loader import center_crop_or_pad, load_series_volume, normalize_intensity, resample_volume


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-root", default=str(ROOT / "data" / "raw" / "train_series"))
    ap.add_argument("--manifest", default=str(ROOT / "data" / "processed" / "manifest" / "series_manifest.csv"))
    ap.add_argument("--cache-root", default=str(ROOT / "data" / "processed" / "volume_cache"))
    ap.add_argument("--target-spacing", nargs=3, type=float, default=[3.0, 0.5, 0.5],
                     help="dz dy dx, mm")
    ap.add_argument("--target-shape", nargs=3, type=int, default=[32, 256, 256],
                     help="D H W voxels, cropped/padded after resampling to bound file size")
    ap.add_argument("--limit", type=int, default=None, help="only the first N series (smoke runs)")
    args = ap.parse_args()
    target_shape = tuple(args.target_shape)

    target_spacing = tuple(args.target_spacing)
    tag = spacing_tag(target_spacing)
    cache_dir = Path(args.cache_root) / tag
    cache_dir.mkdir(parents=True, exist_ok=True)
    print(f"Caching to {cache_dir} (target_spacing={target_spacing})")

    with open(args.manifest, newline="") as f:
        rows = list(csv.DictReader(f))
    if args.limit:
        rows = rows[: args.limit]
    print(f"{len(rows)} series in manifest")

    n_done = n_skipped = n_failed = 0
    for row in rows:
        study, series = row["StudyInstanceUID"], row["SeriesInstanceUID"]
        out_dir = cache_dir / study
        out_path = out_dir / f"{series}.npy"
        meta_path = out_dir / f"{series}.json"
        if out_path.exists() and meta_path.exists():
            n_skipped += 1
            continue
        series_dir = Path(args.raw_root) / study / series
        if not series_dir.is_dir():
            n_failed += 1
            continue  # not downloaded locally -- expected for most of the dataset today
        try:
            result = load_series_volume(series_dir)
            resampled, achieved_spacing = resample_volume(
                result.volume,
                spacing=(result.slice_spacing, result.row_spacing, result.col_spacing),
                target_spacing=target_spacing,
            )
            normalized = normalize_intensity(resampled)
            bounded = center_crop_or_pad(normalized, target_shape)
            out_dir.mkdir(parents=True, exist_ok=True)
            np.save(out_path, bounded)
            with open(meta_path, "w") as f:
                json.dump({
                    "plane": result.plane,
                    "target_spacing": list(target_spacing),
                    "achieved_spacing": list(achieved_spacing),
                    "resampled_shape": list(normalized.shape),
                    "shape": list(bounded.shape),
                    "orientation_consistent": result.orientation_consistent,
                    "uniform_spacing": result.uniform_spacing,
                }, f)
            n_done += 1
        except Exception as e:
            n_failed += 1
            print(f"  FAILED {study}/{series}: {e!r}")

    print(f"\nDone: {n_done} cached, {n_skipped} already cached, {n_failed} skipped/failed "
          f"(missing locally or unreadable)")


if __name__ == "__main__":
    main()
