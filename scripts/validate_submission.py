"""Run before every submission upload -- [redacted: private project notes]: "never burn a submission
slot on a formatting bug." Exits non-zero (and prints every issue) on any
problem, so it's safe to chain with `&&` before an actual upload command.

Usage: python scripts/validate_submission.py path/to/submission.csv
       [--expected path/to/test.csv]   # defaults to data/raw/sample_submission.csv
"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from inference.validate_submission import expected_ids_from_csv, validate_submission_file


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("submission")
    ap.add_argument("--expected", default=str(ROOT / "data" / "raw" / "sample_submission.csv"))
    args = ap.parse_args()

    expected_ids = expected_ids_from_csv(args.expected)
    is_valid, issues = validate_submission_file(args.submission, expected_ids)

    if is_valid:
        print(f"OK: {args.submission} matches {args.expected} ({len(expected_ids)} studies).")
        return
    print(f"INVALID: {args.submission} ({len(issues)} issue(s)):")
    for issue in issues:
        print(f"  - {issue}")
    sys.exit(1)


if __name__ == "__main__":
    main()
