"""Partition the mined-pool studies NOT already in the pilot cache
(data/splits/pilot_study_ids.txt) into fixed-size shards for full-dataset
volume caching, since one Kaggle kernel's output can't hold the whole
~103GB (measured from the 558-study pilot: 13GB / 558 studies =~23.3MB/study;
scales to ~4349 mined studies =~101GB). Gold studies need no further
caching -- already fully covered by the pilot (all 58 are in it).

Shard size 700 -> ~16.3GB/shard (real measured rate), comfortably under the
13GB that already completed cleanly with margin for the unknown exact Kaggle
per-kernel output limit (not confirmed -- [redacted: private project notes] open item).

Usage: python scripts/select_cache_shards.py
Needs: data/processed/cv_folds.npz, data/splits/pilot_study_ids.txt.
Writes: data/splits/cache_shard_{i}_study_ids.txt (committed -- small, like
pilot_study_ids.txt, not the caches themselves).
"""

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

SHARD_SIZE = 700
OUT_DIR = ROOT / "data" / "splits"


def main():
    folds = np.load(ROOT / "data" / "processed" / "cv_folds.npz", allow_pickle=True)
    mined_ids = list(folds["study_ids"])
    pilot_ids = set(
        l.strip() for l in (ROOT / "data" / "splits" / "pilot_study_ids.txt").read_text().splitlines() if l.strip()
    )
    remaining = [s for s in mined_ids if s not in pilot_ids]
    print(f"mined pool: {len(mined_ids)}, already cached (pilot's mined subset): "
          f"{len(mined_ids) - len(remaining)}, remaining to shard: {len(remaining)}")

    n_shards = -(-len(remaining) // SHARD_SIZE)
    for i in range(n_shards):
        shard = remaining[i * SHARD_SIZE:(i + 1) * SHARD_SIZE]
        path = OUT_DIR / f"cache_shard_{i+1}_study_ids.txt"
        path.write_text("\n".join(shard))
        print(f"shard {i+1}/{n_shards}: {len(shard)} studies -> {path}")


if __name__ == "__main__":
    main()
