"""Step 2 of the OA weak-supervision pipeline: build the vote matrix over
all training reports and cache it, per the caching working agreement (never
recompute this from raw text on every run).

Usage: python scripts/build_oa_vote_matrix.py
Reads data/raw/train.csv, writes data/processed/oa_vote_matrix.npz
(StudyInstanceUID array + int8 vote matrix + column names) and prints a
label-free coverage summary (per LF: how often it fires vs. abstains).
"""

import csv
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from data.label_mining.weak_supervision.vote_matrix import (
    build_vote_matrix,
    vote_matrix_summary,
)
from translate_reports import load_or_build_translations


def main():
    train_csv = ROOT / "data" / "raw" / "train.csv"
    with open(train_csv, newline="", encoding="utf-8") as f:
        rows = list(csv.reader(f))
    header, rows = rows[0], rows[1:]

    study_ids = [row[0] for row in rows]
    reports = [row[1] for row in rows]
    print(f"Building vote matrix for {len(reports)} reports...")
    translated_reports = load_or_build_translations(study_ids, reports)

    matrix, columns = build_vote_matrix(reports, translated_reports)

    out_path = ROOT / "data" / "processed" / "oa_vote_matrix.npz"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        out_path,
        study_ids=np.array(study_ids, dtype=object),
        matrix=matrix.astype(np.int8),
        columns=np.array([f"{c}|{lf}" for c, lf in columns], dtype=object),
    )
    print(f"Saved matrix {matrix.shape} to {out_path}")

    print("\nLabel-free coverage summary (no gold labels used):")
    summary = vote_matrix_summary(matrix, columns)
    for (compartment, lf_name), stats in summary.items():
        print(
            f"  {compartment:<12} {lf_name:<18} "
            f"pos={stats['n_positive']:>5} neg={stats['n_negative']:>5} "
            f"abstain={stats['n_abstain']:>5} coverage={stats['coverage']:.1%}"
        )


if __name__ == "__main__":
    main()
