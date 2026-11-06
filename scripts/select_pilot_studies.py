"""Select the pilot subset of studies cached by kaggle_kernels/volume_cache_pilot/
(scripts/build_volume_cache_kernel.py reads its output). 100 studies stratified
across each of the 5 patient-grouped, site-stratified CV folds (scripts/build_cv_folds.py's
output) plus all 58 gold studies, so evaluate_on_gold has real cached data once a model
exists. Fixed seed -- this is meant to be exactly reproducible, not resampled per run.

Usage: python scripts/select_pilot_studies.py
Needs: data/processed/cv_folds.npz (run scripts/build_cv_folds.py first).
Writes: data/splits/pilot_study_ids.txt (one StudyInstanceUID per line, committed --
this is a small, meaningful selection artifact, not a large derived cache, so it lives
in data/splits/ like data/splits/oa_investigation_split.json, not data/processed/).
"""

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

SEED = 0
N_PER_FOLD = 100
OUT_PATH = ROOT / "data" / "splits" / "pilot_study_ids.txt"


def main():
    folds = np.load(ROOT / "data" / "processed" / "cv_folds.npz", allow_pickle=True)
    study_ids = list(folds["study_ids"])
    fold = folds["fold"]
    gold_ids = list(folds["gold_study_ids"])
    n_folds = int(folds["n_folds"])
    print(f"mined pool: {len(study_ids)}, gold: {len(gold_ids)}, n_folds: {n_folds}")

    rng = np.random.default_rng(SEED)
    pilot = []
    for f in range(n_folds):
        idx = np.where(fold == f)[0]
        take = rng.choice(idx, size=min(N_PER_FOLD, len(idx)), replace=False)
        pilot.extend(study_ids[i] for i in take)
    print(f"pilot mined studies: {len(pilot)}")

    pilot_all = sorted(set(pilot) | set(gold_ids))
    print(f"pilot total (mined + gold, deduped): {len(pilot_all)}")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text("\n".join(pilot_all))
    print(f"wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
