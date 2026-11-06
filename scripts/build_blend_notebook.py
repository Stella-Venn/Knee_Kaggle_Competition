"""[redacted: detail about the data challenge this code was written for]

[redacted: note about the author's personal workflow or decisions]
[redacted: detail about the data challenge this code was written for]
fetched into third_party/ (see third_party/README.md). Its first 30 cells (the public
pipeline, ending with the cell that publishes submission.csv) run unchanged. Its last three
[redacted: detail about the data challenge this code was written for]
back to the public result -- are replaced with ours:

  1/3  stash the public submission.csv verbatim
  2/3  run our 5-fold DINOv2 z4 streaming inference as a SUBPROCESS (its own Python process, so
       [redacted: detail about the data challenge this code was written for]
       writing to /kaggle/working/ours_dino/. Our package is passed explicitly: the public
       [redacted: detail about the data challenge this code was written for]
       would otherwise be able to pick up.
  3/3  rank-blend: final = rank((1 - W) * rank(public) + W * rank(ours)), one global W.
       ANY failure in 2/3 or 3/3 restores the public submission verbatim (never worse than the
       [redacted: detail about the data challenge this code was written for]

Blend weights cannot be validated on our OOF (the public models trained on all data), so W is
[redacted: detail about the data challenge this code was written for]

Usage:
  python scripts/build_blend_notebook.py --weight 0.15
  kaggle kernels push -p kaggle_kernels/blend_d4_w15
"""
import argparse
import base64
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE_DIR = ROOT / "third_party" / "pjmathematician_rsna-knee-d4-blend"
BASE_NB = BASE_DIR / "rsna-knee-d4-blend.ipynb"
OWNER = "your-kaggle-username"
N_KEEP = 30  # cells 0..29: the public pipeline through "publish submission.csv"

sys.path.insert(0, str(ROOT / "scripts"))
import build_dinov2_fork as fork  # noqa: E402

ATTRIBUTION = """# RSNA Knee -- public d4-blend + our DINOv2 depth-zone ensemble 

**Base (cells 1-30, unchanged):** the public notebook
[pjmathematician/rsna-knee-d4-blend](https://www.kaggle.com/code/pjmathematician/rsna-knee-d4-blend)
and the public models it attaches (by mattiaangeli, dreaddevelopment, antoinegg1, prvsiyan,
marwanmath, pilkwang, sofiaanjenje and others) -- full credit to those authors.

**Ours (last three cells):** a 5-fold DINOv2-small ensemble with 4 depth zones x 3 slices per
sequence, trained by us (fork of pilkwang/rsna-knee-baseline-v1), run in its own process and
rank-blended with the public result at weight W. Any failure in our part restores the public
submission unchanged.
"""

STASH = r'''# ============ ours 1/3: stash the public submission verbatim ============
import shutil as _sv_shutil
from pathlib import Path as _SvPath
import pandas as _sv_pd
_SV_WORK = _SvPath("/kaggle/working")
_SV_TARGETS = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA",
               "PF OA", "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"]
_SV_SUB, _SV_PUBLIC = _SV_WORK / "submission.csv", _SV_WORK / "_public_d4.csv"
_sv_pub = _sv_pd.read_csv(_SV_SUB, dtype={"StudyInstanceUID": str})
assert _sv_pub.columns.tolist() == ["StudyInstanceUID", *_SV_TARGETS], _sv_pub.columns.tolist()
assert _sv_pub[_SV_TARGETS].notna().all().all(), "public submission has nulls"
_sv_shutil.copy2(_SV_SUB, _SV_PUBLIC)
print(f"[ours 1/3] public submission stashed: {_sv_pub.shape}", flush=True)
'''

LEG = r'''# ============ ours 2/3: our DINOv2 z4 ensemble, in its own process ============
import base64 as _sv_b64, glob as _sv_glob, os as _sv_os, subprocess as _sv_sp, time as _sv_t
_SV_LEG_OUT = _SV_WORK / "ours_dino"
_SV_LEG_OUT.mkdir(parents=True, exist_ok=True)
_sv_script = _SvPath("/kaggle/tmp/ours_leg.py")
_sv_script.parent.mkdir(parents=True, exist_ok=True)
_sv_script.write_bytes(_sv_b64.b64decode(_SV_LEG_B64))

def _sv_find_package():
    # Known mount points first, then a shallow glob. (v1 used a recursive glob over every
    # attached dataset: 6 minutes on this notebook's ~15 large mounts.)
    for d in ["/kaggle/input/datasets/your-kaggle-username/rsna-knee-dinov2-z4-5fold",
              "/kaggle/input/rsna-knee-dinov2-z4-5fold"]:
        if (_SvPath(d) / "manifest.json").is_file() and (_SvPath(d) / "r336_z4_f0.pt").is_file():
            return d
    for pat in ["/kaggle/input/*/r336_z4_f0.pt", "/kaggle/input/*/*/r336_z4_f0.pt",
                "/kaggle/input/*/*/*/r336_z4_f0.pt"]:
        hits = [p for p in _sv_glob.glob(pat) if (_SvPath(p).parent / "manifest.json").is_file()]
        if hits:
            return str(_SvPath(hits[0]).parent)
    return None

def _sv_proc_age_h():
    up = float(open("/proc/uptime").read().split()[0])
    start = int(open("/proc/self/stat").read().split(")")[-1].split()[19]) / _sv_os.sysconf("SC_CLK_TCK")
    return (up - start) / 3600

_SV_PKG = _sv_find_package()
_SV_LEFT_H = 9.0 - _sv_proc_age_h()
print(f"[ours 2/3] package={_SV_PKG} budget left {_SV_LEFT_H:.2f} h", flush=True)
_sv_leg_rc = None
if _SV_PKG is None:
    print("[ours 2/3] SOFT FAIL: our package is not attached -> public only", flush=True)
elif _SV_LEFT_H < 1.5:
    print("[ours 2/3] SOFT FAIL: < 1.5 h left -> public only", flush=True)
else:
    import torch as _sv_torch
    # No progress bars, and the leg's output goes to a FILE, not the notebook's inherited
    # pipes: v1 hung for 8 h right after its first model loaded -- the model-loading progress
    # bars filled a pipe that nothing was draining while this cell waited (classic pipe
    # deadlock). The log's tail is printed here once the leg ends.
    _sv_env = dict(_sv_os.environ, FORK_PACKAGE=_SV_PKG, PYTHONUNBUFFERED="1",
                   TRANSFORMERS_VERBOSITY="error", HF_HUB_DISABLE_PROGRESS_BARS="1",
                   TQDM_DISABLE="1")
    if _sv_torch.cuda.device_count() >= 2:  # the public pipeline ran on GPU 0; use a fresh one
        _sv_env["CUDA_VISIBLE_DEVICES"] = str(_sv_torch.cuda.device_count() - 1)
    _SV_LEG_LOG = _SV_LEG_OUT / "leg.log"
    _sv_t0 = _sv_t.time()
    try:
        with open(_SV_LEG_LOG, "w") as _sv_logf:
            _sv_leg_rc = _sv_sp.run(["python", str(_sv_script)], cwd=str(_SV_LEG_OUT), env=_sv_env,
                                    stdin=_sv_sp.DEVNULL, stdout=_sv_logf, stderr=_sv_sp.STDOUT,
                                    timeout=int(min(_SV_LEFT_H - 0.5, 3.0) * 3600)).returncode  # ~25 min expected at ~1,300 studies
    except Exception as _sv_e:
        print(f"[ours 2/3] SOFT FAIL: {type(_sv_e).__name__}: {_sv_e}", flush=True)
    print(f"[ours 2/3] leg rc={_sv_leg_rc}, {(_sv_t.time() - _sv_t0) / 60:.1f} min; "
          f"last lines of {_SV_LEG_LOG}:", flush=True)
    try:
        print("".join(open(_SV_LEG_LOG, errors="replace").readlines()[-40:]), flush=True)
    except Exception as _sv_e:
        print(f"[ours 2/3] could not read the leg log: {_sv_e}", flush=True)
'''

BLEND = r'''# ============ ours 3/3: rank blend, public restored on any failure ============
import numpy as _sv_np
_SV_W = SV_BLEND_WEIGHT
_SV_OURS = _SV_LEG_OUT / "submission.csv"

def _sv_rank(df):
    return df[_SV_TARGETS].rank(method="average", pct=True)

try:
    if _sv_leg_rc != 0:
        raise RuntimeError(f"our leg did not finish cleanly (rc={_sv_leg_rc})")
    _sv_ours = _sv_pd.read_csv(_SV_OURS, dtype={"StudyInstanceUID": str})
    if _sv_ours.columns.tolist() != ["StudyInstanceUID", *_SV_TARGETS]:
        raise RuntimeError(f"our schema drift: {_sv_ours.columns.tolist()}")
    _sv_ours = (_sv_ours.set_index("StudyInstanceUID")
                .reindex(_sv_pub["StudyInstanceUID"]).reset_index())
    if _sv_ours[_SV_TARGETS].isna().any().any():
        raise RuntimeError("our leg is missing studies or has nulls")
    _sv_sd = _sv_ours[_SV_TARGETS].to_numpy(float).std(axis=0)
    if len(_sv_pub) > 10 and float(_sv_sd.min()) < 1e-9:
        raise RuntimeError("our leg has a constant column (its 0.5 fallback fired)")
    _sv_out = _sv_pub.copy()
    _sv_out[_SV_TARGETS] = (1.0 - _SV_W) * _sv_rank(_sv_pub).to_numpy() + _SV_W * _sv_rank(_sv_ours).to_numpy()
    _sv_out[_SV_TARGETS] = _sv_out[_SV_TARGETS].rank(method="average", pct=True)
    assert _sv_np.isfinite(_sv_out[_SV_TARGETS].to_numpy(float)).all()
    _sv_out.to_csv(_SV_SUB, index=False)
    _sv_corr = float(_sv_np.nanmean([_sv_rank(_sv_pub).iloc[:, j].corr(_sv_rank(_sv_ours).iloc[:, j], method="spearman")
                                     for j in range(len(_SV_TARGETS))]))
    print(f"[ours 3/3] BLENDED: public {1 - _SV_W:.2f} + ours {_SV_W:.2f}; "
          f"mean per-label rank corr public vs ours {_sv_corr:.4f}", flush=True)
except Exception as _sv_exc:
    _sv_shutil.copy2(_SV_PUBLIC, _SV_SUB)
    print(f"[ours 3/3] NOT blended ({type(_sv_exc).__name__}: {_sv_exc}); "
          f"public submission restored", flush=True)

_sv_final = _sv_pd.read_csv(_SV_SUB, dtype={"StudyInstanceUID": str})
assert _sv_final.columns.tolist() == ["StudyInstanceUID", *_SV_TARGETS]
assert _sv_final[_SV_TARGETS].notna().all().all() and len(_sv_final) == len(_sv_pub)
print(f"[ours] submission.csv verified: {_sv_final.shape}", flush=True)
'''


def leg_script(package):
    """Our infer-mode fork notebook as one script, writing into its cwd-relative dir."""
    cfg = {"mode": "infer", "fold": 0, "zones": 1, "shard": 0, "n_shards": 3, "epochs": 10,
           "batch": 8, "n_rehearse": 1300}
    nb = fork.build(cfg)
    code = "\n\n".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code")
    code = "\n".join(l for l in code.splitlines() if not l.lstrip().startswith(("%", "!"))
                     and l.strip() != "from __future__ import annotations")
    code = "from __future__ import annotations\n" + code
    assert code.count("/kaggle/working") > 0
    code = code.replace("/kaggle/working", "/kaggle/working/ours_dino")
    old = "    pkg = find_weights()\n"
    assert code.count(old) == 1, "find_weights call site moved"
    code = code.replace(old, '    pkg = Path(os.environ["FORK_PACKAGE"])  # your-kaggle-username blend: '
                             'our package, never another attached manifest\n')
    compile(code, "ours_leg.py", "exec")
    return code


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weight", type=float, required=True, help="our rank weight W in [0, 1]; 1.0 = ours only (diagnostic: shows whether our leg runs in scored runs)")
    ap.add_argument("--package", default=f"{OWNER}/rsna-knee-dinov2-z4-5fold")
    args = ap.parse_args()
    assert 0 <= args.weight <= 1

    nb = json.loads(BASE_NB.read_text())
    cells = nb["cells"]
    publish = "".join(cells[N_KEEP - 1]["source"])
    assert "Publish submission.csv only after" in publish, "base notebook changed: cell 29"
    assert all("OUR LEG" in "".join(c["source"]) for c in cells[N_KEEP:]), "base tail changed"
    leg_b64 = base64.b64encode(leg_script(args.package).encode()).decode()

    def code_cell(src):
        return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
                "source": src.splitlines(keepends=True)}
    kept = cells[:N_KEEP]
    for c in kept:
        if c["cell_type"] == "code":
            c["outputs"], c["execution_count"] = [], None
    nb["cells"] = ([{"cell_type": "markdown", "metadata": {}, "source": ATTRIBUTION.splitlines(keepends=True)}]
                   + kept
                   + [code_cell(STASH),
                      code_cell(f'_SV_LEG_B64 = "{leg_b64}"\n' + LEG),
                      code_cell(BLEND.replace("SV_BLEND_WEIGHT", repr(args.weight)))])
    for i, c in enumerate(nb["cells"][-3:]):
        compile("".join(c["source"]), f"ours_cell_{i}", "exec")

    base_meta = json.loads((BASE_DIR / "kernel-metadata.json").read_text())
    tag = f"w{round(args.weight * 100):02d}"
    out = ROOT / "kaggle_kernels" / f"blend_d4_{tag}"
    meta = {
        "id": f"{OWNER}/rsna-knee-blend-d4-{tag}",
        "title": f"RSNA Knee - Blend D4 {tag.upper()}",
        "code_file": "blend.ipynb",
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": True,
        "enable_internet": False,
        "docker_image": base_meta["docker_image"],
        "machine_shape": base_meta.get("machine_shape", "NvidiaTeslaT4"),
        "dataset_sources": [d for d in base_meta["dataset_sources"] if d] + [args.package],
        "model_sources": base_meta["model_sources"],
        "competition_sources": ["rsna-knee-abnormality-detection"],
        "kernel_sources": base_meta["kernel_sources"],
    }
    out.mkdir(parents=True, exist_ok=True)
    (out / "blend.ipynb").write_text(json.dumps(nb, indent=1))
    (out / "kernel-metadata.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(f"wrote {out.relative_to(ROOT)} ({meta['id']}, W={args.weight}, "
          f"{len(meta['dataset_sources'])} datasets)")


if __name__ == "__main__":
    main()
