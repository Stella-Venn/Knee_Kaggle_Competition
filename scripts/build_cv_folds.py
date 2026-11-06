"""Build the patient-grouped, site-stratified CV folds over the mined-label pool
(every training study except the gold 58, which are reserved for evaluate_on_gold).

Needs: data/raw/train.csv, data/processed/manifest/study_manifest.csv (run the
dicom_manifest Kaggle kernel first, see scripts/build_manifest_kernel.py).
Writes: data/processed/cv_folds.npz (study_ids, fold, site_proxy; gold studies
excluded -- look them up from train.csv directly, they don't need a fold).
"""

import csv
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from data.splits import make_cv_folds, site_proxy_key
from utils.constants import LABEL_COLUMNS

N_FOLDS = 5
SEED = 0


def main():
    with open(ROOT / "data" / "raw" / "train.csv", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    gold_ids = {r["StudyInstanceUID"] for r in rows if any(r[c] != "" for c in LABEL_COLUMNS)}
    mined_ids = [r["StudyInstanceUID"] for r in rows if r["StudyInstanceUID"] not in gold_ids]
    print(f"{len(rows)} total studies, {len(gold_ids)} gold (excluded from folds), "
          f"{len(mined_ids)} in the mined-label pool")

    with open(ROOT / "data" / "processed" / "manifest" / "study_manifest.csv", newline="") as f:
        manifest = {r["StudyInstanceUID"]: r for r in csv.DictReader(f)}
    missing = [sid for sid in mined_ids if sid not in manifest]
    if missing:
        raise SystemExit(f"{len(missing)} mined studies missing from the manifest, e.g. {missing[:3]}")

    patient_ids = [manifest[sid]["PatientID"] or sid for sid in mined_ids]
    site_proxies = [
        site_proxy_key(manifest[sid]["Manufacturer"], manifest[sid]["MagneticFieldStrength"])
        for sid in mined_ids
    ]
    fold = make_cv_folds(mined_ids, patient_ids, site_proxies, n_folds=N_FOLDS, seed=SEED)

    out_path = ROOT / "data" / "processed" / "cv_folds.npz"
    np.savez(
        out_path,
        study_ids=np.array(mined_ids, dtype=object),
        fold=fold,
        site_proxy=np.array(site_proxies, dtype=object),
        gold_study_ids=np.array(sorted(gold_ids), dtype=object),
        n_folds=N_FOLDS,
        seed=SEED,
    )
    print(f"\nSaved {len(mined_ids)}-study fold assignment to {out_path}")

    print(f"\n{'fold':<6}{'n_studies':>11}   site_proxy breakdown (top 4)")
    import collections
    for f in range(N_FOLDS):
        idx = np.where(fold == f)[0]
        top = collections.Counter(site_proxies[i] for i in idx).most_common(4)
        print(f"{f:<6}{len(idx):>11}   {top}")


if __name__ == "__main__":
    main()
