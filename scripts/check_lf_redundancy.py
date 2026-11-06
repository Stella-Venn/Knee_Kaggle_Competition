"""Step 4 of the OA weak-supervision pipeline: pairwise LF redundancy check
against the real, full 4407-report vote matrix. Label-free, same as Step 2 --
answers whether LFs are restating each other, not whether they're correct.

Usage: python scripts/check_lf_redundancy.py
Requires data/processed/oa_vote_matrix.npz (run build_oa_vote_matrix.py first).
"""

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from data.label_mining.weak_supervision.redundancy import pairwise_agreement
from data.label_mining.weak_supervision.vote_matrix import VOTE_COLUMNS


def main():
    cache_path = ROOT / "data" / "processed" / "oa_vote_matrix.npz"
    cached = np.load(cache_path, allow_pickle=True)
    matrix = cached["matrix"].astype(int)
    print(f"Loaded vote matrix {matrix.shape}")

    if matrix.shape[1] != len(VOTE_COLUMNS):
        raise SystemExit(
            f"Cached matrix has {matrix.shape[1]} columns but VOTE_COLUMNS has "
            f"{len(VOTE_COLUMNS)} -- stale cache, rerun build_oa_vote_matrix.py first."
        )

    results = pairwise_agreement(matrix, VOTE_COLUMNS)

    print(f"\n{'Compartment':<12} {'LF pair':<55} {'n_both':>7} {'overlap':>8} "
          f"{'agreement':>10}  verdict")
    for r in results:
        pair_str = f"{r['lf_a']} / {r['lf_b']}"
        agree_str = f"{r['agreement']:.1%}" if r["agreement"] is not None else "n/a"
        overlap_str = f"{r['coverage_overlap']:.1%}" if r["coverage_overlap"] is not None else "n/a"
        print(f"{r['compartment']:<12} {pair_str:<55} {r['n_both_fired']:>7} "
              f"{overlap_str:>8} {agree_str:>10}  {r['verdict']}")


if __name__ == "__main__":
    main()
