"""Local end-to-end smoke test of the GENERATED DINOv2 fork notebooks, on CPU.

Stage `train`: generates the train-mode notebook (2 depth zones, 2 shards, 1 epoch) and
runs it against a fake pixel cache of 24 real training studies (random pixels, real ids
and labels, gold included) -> writes a package member + OOF predictions.
Stage `infer`: generates the infer-mode notebook with that package attached and runs it
on the local test DICOMs -> reloads the member, checks its fingerprint, writes a
submission. Only Kaggle paths are redirected and the image size shrunk to 112 px.

Run after any change to scripts/build_dinov2_fork.py, before pushing to Kaggle:
  python scripts/smoke_dinov2_fork.py train && python scripts/smoke_dinov2_fork.py infer
Inputs are fetched on first use: the public LLM label table (kaggle CLI) and
facebook/dinov2-small (Hugging Face). Work dir: $FORK_SMOKE_DIR (default /tmp/rsna_fork_smoke).
NOTE: regenerates kaggle_kernels/dinov2_infer with a dummy package id and writes
kaggle_kernels/dinov2_train_z2_f0 -- regenerate/delete those before committing.
"""
import hashlib, json, os, subprocess, sys
from pathlib import Path
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
SM = Path(os.environ.get("FORK_SMOKE_DIR", "/tmp/rsna_fork_smoke"))
INP, WORK = SM / "input", SM / "working"
COMP = INP / "competitions" / "rsna-knee-abnormality-detection"
PY = sys.executable
IMG, ZONES, GROUP, N_SHARDS = 112, 2, 3, 2
SLOTS = ["SAG_FLUID_FS", "COR_FLUID_FS", "AX_FLUID_FS", "SAG_FLUID_NOFS", "COR_T1", "SAG_T1"]
RULES = {"decode_fill": "nearest", "lat": "centre", "order": "normal", "slot_fallback": False}


def setup():
    COMP.mkdir(parents=True, exist_ok=True)
    for f in ["train.csv", "test.csv", "train_series.csv", "test_series.csv", "sample_submission.csv"]:
        if not (COMP / f).exists():
            (COMP / f).symlink_to(REPO / "data" / "raw" / f)
    # test_series: every test.csv id links to the one test study with local DICOMs, so
    # inference has 3 real studies (enough for several chunks at INFER_CHUNK = 2)
    ts = COMP / "test_series"
    ts.mkdir(exist_ok=True)
    local = next((REPO / "data" / "raw" / "test_series").iterdir())
    for sid in pd.read_csv(REPO / "data" / "raw" / "test.csv")["StudyInstanceUID"]:
        if not (ts / sid).exists():
            (ts / sid).symlink_to(local)
    lab = INP / "datasets" / "pilkwang" / "rsna-knee-llm-labels"
    lab.mkdir(parents=True, exist_ok=True)
    if not (lab / "report_labels_v2.csv").exists():
        subprocess.run(["kaggle", "datasets", "download", "pilkwang/rsna-knee-llm-labels",
                        "-p", str(lab), "--unzip"], check=True)
    dino = INP / "models" / "metaresearch" / "dinov2" / "pytorch" / "small" / "1"
    if not (dino / "config.json").exists():
        from huggingface_hub import snapshot_download
        snapshot_download("facebook/dinov2-small", local_dir=str(dino),
                          allow_patterns=["config.json", "*.safetensors", "preprocessor_config.json"])
    # fake pixel cache: 24 real training studies incl. gold, random pixels, 2 shards
    tr = pd.read_csv(REPO / "data" / "raw" / "train.csv")
    gold = tr[tr["ACL"].notna()]["StudyInstanceUID"].tolist()[:8]
    other = tr[tr["ACL"].isna()]["StudyInstanceUID"].tolist()[:16]
    studies = sorted(gold + other)
    key = {"img": IMG, "group": GROUP, "slices": GROUP * ZONES, "crop_mm": 130.0,
           "band": [0.2, 0.8], "rules": RULES, "slots": SLOTS}
    rng = np.random.default_rng(0)
    for k in range(N_SHARDS):
        mine = [s for i, s in enumerate(studies) if i % N_SHARDS == k]
        d = INP / "notebooks" / "your-kaggle-username" / f"rsna-knee-dinov2-cache-z{ZONES}-s{k}" / "pixel_cache"
        d.mkdir(parents=True, exist_ok=True)
        np.save(d / f"cache_{k}.npy", rng.integers(0, 256, (len(mine), 6, GROUP * ZONES, IMG, IMG), dtype=np.uint8))
        m = np.ones((len(mine), 6), np.float32); m[:, 5] = 0  # one slot absent everywhere
        np.save(d / f"mask_{k}.npy", m)
        (d / f"meta_{k}.json").write_text(json.dumps({"shard": k, "n_shards": N_SHARDS,
                                                      "studies": mine, "pixel_key": key, "decode_failed": 0}))
    rep = tr.set_index("StudyInstanceUID")["Report"].fillna("")
    folds = [int(hashlib.md5(rep[s].encode()).hexdigest()[:8], 16) % 5 for s in studies]
    print("fold counts:", {f: folds.count(f) for f in range(5)})


def notebook_to_script(nb_path, out_path, work):
    nb = json.loads(Path(nb_path).read_text())
    code = "\n\n".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code")
    code = "\n".join(l for l in code.splitlines() if not l.lstrip().startswith(("%", "!"))
                     and l.strip() != "from __future__ import annotations")
    code = "from __future__ import annotations\n" + code  # IPython carries the flag across cells
    code = code.replace('"/kaggle/input', f'"{INP}').replace("/kaggle/working", str(work))
    code = code.replace("CACHE_IMG = 336", f"CACHE_IMG = {IMG}").replace("INFER_CHUNK = 256", "INFER_CHUNK = 2").replace('{"name": "r336", "img": 336}', f'{{"name": "r336", "img": {IMG}}}')
    assert '"/kaggle/' not in code, "unredirected kaggle path"
    Path(out_path).write_text(code)


def run(script, cwd):
    cwd.mkdir(parents=True, exist_ok=True)
    r = subprocess.run([PY, script], cwd=cwd, capture_output=True, text=True, timeout=3000)
    print(r.stdout[-6000:]); print(r.stderr[-4000:])
    return r.returncode


if __name__ == "__main__":
    setup()
    stage = sys.argv[1]
    if stage == "train":
        # SMOKE_BACKBONE="timm:<name>@<px>" smokes idea 4's timm path (tag "smk"); the member
        # is added to the same package, so a later `infer` runs a mixed-backbone ensemble.
        bb = os.environ.get("SMOKE_BACKBONE")
        extra = ["--backbone", bb, "--lr-bb", "1e-4", "--tag", "smk"] if bb else []
        subprocess.run([PY, str(REPO / "scripts" / "build_dinov2_fork.py"), "--mode", "train", "--fold", "0",
                        "--zones", str(ZONES), "--n-shards", str(N_SHARDS), "--epochs", "1", "--batch", "2"]
                       + extra, check=True)
        kdir = f"dinov2_train_z{ZONES}_f0" + ("_smk" if bb else "")
        notebook_to_script(REPO / "kaggle_kernels" / kdir / "dinov2_fork.ipynb", SM / "train_smoke.py", WORK / "train")
        sys.exit(run(SM / "train_smoke.py", WORK / "train"))
    if stage == "infer":
        pkg = INP / "datasets" / "your-kaggle-username" / "smoke-package"
        pkg.mkdir(parents=True, exist_ok=True)
        for f in (WORK / "train" / "package").iterdir():
            if not (pkg / f.name).exists():
                (pkg / f.name).symlink_to(f)
        subprocess.run([PY, str(REPO / "scripts" / "build_dinov2_fork.py"), "--mode", "infer", "--package", "x/smoke"], check=True)
        notebook_to_script(REPO / "kaggle_kernels" / "dinov2_infer" / "dinov2_fork.ipynb", SM / "infer_smoke.py", WORK / "infer")
        sys.exit(run(SM / "infer_smoke.py", WORK / "infer"))
