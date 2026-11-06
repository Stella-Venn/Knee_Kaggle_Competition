"""Answer the manifest-related open questions in [redacted: private project notes] from the series/study
manifest CSVs produced by the dicom_manifest Kaggle kernel.

Usage: python scripts/analyze_manifest.py [manifest_dir]
  manifest_dir defaults to data/processed/manifest (series_manifest.csv,
  study_manifest.csv, errors.json).
"""

import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def pct(n, d):
    return f"{n}/{d} ({n / d:.1%})" if d else "0/0"


def main():
    d = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "data" / "processed" / "manifest"
    series = load(d / "series_manifest.csv")
    studies = load(d / "study_manifest.csv")
    errors = json.load(open(d / "errors.json"))
    train = [r for r in series if r["split"] == "train_series"]
    train_ids = {r["StudyInstanceUID"] for r in train}
    train_studies = [s for s in studies if s["StudyInstanceUID"] in train_ids]
    print(f"{len(series)} series ({len(train)} train), {len(studies)} studies "
          f"({len(train_studies)} train), {len(errors)} read errors\n")

    print("== Label alignment: CSV Anatomical_Plane vs plane from DICOM geometry")
    agree = sum(r["csv_Anatomical_Plane"] == r["plane_from_geometry"] for r in train)
    print("  agree:", pct(agree, len(train)))
    for r in [r for r in train if r["csv_Anatomical_Plane"] != r["plane_from_geometry"]][:5]:
        print("   mismatch:", r["SeriesInstanceUID"][-8:], r["csv_Anatomical_Plane"], "vs", r["plane_from_geometry"],
              r["SeriesDescription"])

    print("\n== PatientID -> StudyInstanceUID (is patient-grouped splitting a real leakage guard?)")
    by_patient = defaultdict(set)
    for s in train_studies:
        by_patient[s["PatientID"]].add(s["StudyInstanceUID"])
    multi = {p: v for p, v in by_patient.items() if len(v) > 1}
    print(f"  {len(by_patient)} distinct PatientIDs over {len(train_studies)} studies; "
          f"{len(multi)} PatientIDs span >1 study "
          f"({sum(len(v) for v in multi.values())} studies involved); "
          f"missing PatientID: {sum(1 for s in train_studies if not s['PatientID'])}")
    print("  studies with >1 PatientID inside them:", sum(int(s["n_patient_ids"]) > 1 for s in train_studies))

    print("\n== Series regularity (train)")
    for key in ("uniform_spacing", "instance_order_matches_position", "consistent_shape",
                "consistent_pixel_spacing", "consistent_orientation", "consistent_patient",
                "multiframe", "has_geometry"):
        c = Counter(r[key] for r in train)
        print(f"  {key:<34} True={c['True']} False={c['False']} blank={c['']}")
    n = [int(r["n_slices"]) for r in train]
    print(f"  slices/series: min {min(n)}, median {sorted(n)[len(n)//2]}, max {max(n)}; "
          f"single-slice series: {sum(x == 1 for x in n)}")

    print("\n== Laterality (does it vary? matters for flip augmentation)")
    print("  series-level:", dict(Counter(r["Laterality"] or "<empty>" for r in train)))
    print("  ImageLaterality:", dict(Counter(r["ImageLaterality"] or "<empty>" for r in train)))
    print("  study-level (majority):", dict(Counter(s["Laterality"] or "<empty>" for s in train_studies)))
    mixed = sum(1 for s in train_studies if len({r["Laterality"] for r in train
                if r["StudyInstanceUID"] == s["StudyInstanceUID"] and r["Laterality"]}) > 1) if len(train) < 5000 else "skipped"
    print("  studies whose series disagree on side:", mixed)

    print("\n== Scanner (site proxy)")
    print("  Manufacturer values:", dict(Counter(s["Manufacturer"] or "<empty>" for s in train_studies).most_common()))
    print("  Field strength:", dict(Counter(s["MagneticFieldStrength"] or "<empty>" for s in train_studies)))
    print("  studies with scanner differing across series:", sum(s["scanner_consistent"] == "False" for s in train_studies))
    models = Counter((s["Manufacturer"], s["ManufacturerModelName"]) for s in train_studies)
    print(f"  distinct (Manufacturer, Model): {len(models)}; top 10:")
    for (m, mod), c in models.most_common(10):
        print(f"    {c:>5}  {m} / {mod}")

    print("\n== Image size / bit depth (train series)")
    print("  Rows x Cols top:", Counter(f"{r['Rows']}x{r['Columns']}" for r in train).most_common(8))
    print("  PixelSpacing (row) min/median/max:", end=" ")
    ps = sorted(float(r["PixelSpacingRow"]) for r in train if r["PixelSpacingRow"])
    print(f"{ps[0]:.3f}/{ps[len(ps)//2]:.3f}/{ps[-1]:.3f}")
    print("  BitsStored:", dict(Counter(r["BitsStored"] for r in train)))
    print("  PhotometricInterpretation:", dict(Counter(r["PhotometricInterpretation"] for r in train)))

    if errors:
        print("\n== Read errors (first 5)")
        for e in errors[:5]:
            print("  ", e)


if __name__ == "__main__":
    main()
