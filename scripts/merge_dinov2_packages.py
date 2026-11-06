"""Merge the weights packages written by several DINOv2 fork runs (e.g. 5 folds) into
one package directory that upstream's infer_from_package() reads as an ensemble.

Each input is a directory holding manifest.json + member .pt files (a fold run's
downloaded /kaggle/working/package). Members are combined by id; an id appearing twice
is an error (two runs of the same fold/config -- pick one explicitly). OOF CSVs are
copied along so the package documents its own evaluation.

Usage: python scripts/merge_dinov2_packages.py --out data/runs/dinov2_z4_5fold <pkg_dir> [...]
Then upload --out as a new version of the package dataset.
"""

import argparse
import json
import shutil
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("packages", nargs="+")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    members, seen = [], {}
    for d in map(Path, args.packages):
        man = json.loads((d / "manifest.json").read_text())
        for m in man["members"]:
            if m["id"] in seen:
                raise SystemExit(f"member {m['id']} in both {seen[m['id']]} and {d}")
            seen[m["id"]] = d
            shutil.copy2(d / m["file"], out / m["file"])
            members.append(m)
        for f in d.glob("oof_f*.csv"):
            shutil.copy2(f, out / f.name)
    keys = {m["pixel_group"] for m in members}
    (out / "manifest.json").write_text(json.dumps({"members": members}, indent=2))
    print(f"{len(members)} members -> {out} ({len(keys)} pixel group(s): each group is "
          f"decoded once at inference); folds {sorted(m['fold'] for m in members)}")


if __name__ == "__main__":
    main()
