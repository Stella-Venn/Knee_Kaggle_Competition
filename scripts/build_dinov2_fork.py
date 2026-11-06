"""[redacted: detail about the data challenge this code was written for]

Reads the verbatim upstream copy in third_party/pilkwang_baseline_v1/ and applies an
asserted set of patches -- each one fails loudly if upstream's text is not exactly what
the patch expects, rather than silently patching the wrong thing. A config cell at the
top (`FORK = {...}`) selects the mode:

  legacy  upstream's own path: decode everything, train one 336 px model, save it as a
          weights package (/kaggle/working/package), predict the test set.
  cache   idea 1 ([redacted: private project notes], 2026-09-25): order + decode the TRAINING pixels for
          one shard of the studies and save them (/kaggle/working/pixel_cache/), then
          stop. CPU-only, no GPU quota. Every later training run reads these instead of
          spending ~43 min re-reading DICOM.
  train   ideas 2 + 3: load the cached shards (memory-mapped: RAM is not the limit), train
          ONE fold (report-hash split, fold k of 5), save that member and its out-of-fold
          (OOF) predictions into the package. Skips the test set entirely.
  infer   attach a weights package; upstream's main() takes its inference-only path.

Idea 3 (depth zones): with zones = Z > 1 each slot is read as Z groups of 3 slices spread
over the stack (upstream's read_slot and group layout, unchanged), and EVERY zone goes
through the encoder every step. The head's per-diagnosis attention runs over
(slot, zone) tokens with a learned zone embedding. With Z = 1 the model, its state dict
and its training are exactly upstream's, so 1-zone packages (e.g. our v1) still load.
Upstream's own multi-group option instead trained on one random group per step, which
with study-level labels means often training on a zone that does not contain the finding.

Everything else -- labels (public LLM label table + upstream's rule extractor), DINOv2
fine-tuning, slot choice, laterality normalisation, augmentation -- is upstream's.

Usage:
  python scripts/build_dinov2_fork.py                                 # legacy
  python scripts/build_dinov2_fork.py --mode cache --shard 0 --n-shards 3 --zones 4
  python scripts/build_dinov2_fork.py --mode train --fold 0 --zones 4 --n-shards 3
  [redacted: detail about the data challenge this code was written for]
Then: kaggle kernels push -p kaggle_kernels/<printed dir>
"""

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
UPSTREAM = ROOT / "third_party" / "pilkwang_baseline_v1"
OWNER = "your-kaggle-username"
LABELS_DATASET = "pilkwang/rsna-knee-llm-labels"

ATTRIBUTION = """# RSNA Knee - DINOv2 fork 

Fork of the public notebook **[pilkwang/rsna-knee-baseline-v1](https://www.kaggle.com/code/pilkwang/rsna-knee-baseline-v1)**
by Pilkwang Kim. The method is upstream's; our changes are applied as asserted patches by
`scripts/build_dinov2_fork.py` in our repo. The `FORK` cell below selects the mode:
`legacy` (upstream's path, 336 px, saves a weights package), `cache` (decode one shard of the
training pixels once and save them), `train` (one fold from the cached pixels, optional depth
zones, saves the member and its out-of-fold predictions), `infer` (predict from a package).
"""

# --------------------------------------------------------------------------- #
# Idea 4 ([redacted: private project notes], 2026-09-25): a second backbone family for ensemble diversity.
# A member's config "variant" is either a DINOv2 size ("small", upstream) or
# "timm:<model name>@<input px>". The timm encoder is wrapped so it returns the same
# last_hidden_state layout the head already reads: token 0 is a pseudo-CLS (the max over
# positions), the rest are the feature-map positions -- so the "cls_mean" pool gives
# [max, mean] of the final feature map. The pixel pipeline stays at the cache's 336 px;
# the adapter resamples to the backbone's own input size inside the model, so the member's
# pixel key (and therefore test-time decoding) is unchanged. Pretrained weights are
# downloaded only when training (internet on); inference builds the architecture without
# weights and loads the member's state dict (internet off at scoring).
# --------------------------------------------------------------------------- #
TIMM_BACKBONE = '''class TimmTokens(nn.Module):
    """fork (idea 4): timm feature map -> [pseudo-CLS, position tokens]."""

    def __init__(self, m, img):
        super().__init__()
        self.m = m
        self.img = int(img)

    def forward(self, pixel_values):
        from types import SimpleNamespace
        x = pixel_values
        if x.shape[-1] != self.img:
            x = F.interpolate(x, size=(self.img, self.img), mode="bilinear",
                              align_corners=False)
        f = self.m.forward_features(x)
        tok = f.flatten(2).transpose(1, 2) if f.ndim == 4 else f
        return SimpleNamespace(last_hidden_state=torch.cat([tok.amax(1, keepdim=True), tok], 1))


def build_timm_model(variant, pool="cls_mean", prior=False, zones=1):
    """fork (idea 4): variant = "timm:<name>@<px>"; every backbone parameter trains."""
    import timm
    name, img = variant[len("timm:"):].rsplit("@", 1)
    img = int(img)
    pretrained = FORK["mode"] in ("train", "legacy")  # scoring runs load the member's weights
    try:
        m = timm.create_model(name, pretrained=pretrained, num_classes=0, img_size=img)
    except TypeError:  # CNNs take no img_size
        m = timm.create_model(name, pretrained=pretrained, num_classes=0)
    if FORK.get("grad_ckpt") and FORK["mode"] == "train":
        m.set_grad_checkpointing(True)
    n = sum(p.numel() for p in m.parameters())
    log(f"backbone: timm {name} @{img}px ({'pretrained' if pretrained else 'architecture only'}), "
        f"{n / 1e6:.1f}M params all trainable, feature dim {m.num_features * POOL_PARTS[pool]}")
    return Model(TimmTokens(m, img), m.num_features, pool=pool, prior=prior, zones=zones)


'''

# --------------------------------------------------------------------------- #
# Patches: (old, new) -- exact, unique text in upstream's code cells.
# --------------------------------------------------------------------------- #
PATCHES = [
    # legacy: only the 336 px run
    ('''RUNS = [
    {"name": "r224", "img": 224},
    {"name": "r336", "img": 336},
]''', '''RUNS = [
    # fork: 336 px only -- see scripts/build_dinov2_fork.py
    {"name": "r336", "img": 336},
]'''),
    # epochs / batch from the FORK cell
    ('''EPOCHS = 10
BATCH_STUDIES = 8 ''', '''EPOCHS = FORK["epochs"]         # fork: from the FORK cell (upstream: 10)
BATCH_STUDIES = FORK["batch"]  # fork: from the FORK cell (upstream: 8);'''),
    # cache/train modes: the group count is the zone count, not the memory planner's
    ('''N_GROUP = plan_cache(len(pd.read_csv(ROOT / "train.csv")),
                     len(pd.read_csv(ROOT / "test.csv")))''',
     '''N_GROUP = (ZONES if FORK["mode"] in ("cache", "train") else  # fork: zones fix the layout
           plan_cache(len(pd.read_csv(ROOT / "train.csv")),
                      len(pd.read_csv(ROOT / "test.csv"))))'''),
    # --- idea 3: (slot, zone) tokens in the head ---
    ('''    def __init__(self, dim, n_slot, n_out, hidden=256, p=0.2, prior=False):
        super().__init__()
        self.proj = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, hidden), nn.GELU())
        self.slot_emb = nn.Parameter(torch.randn(n_slot, hidden) * 0.02)''',
     '''    def __init__(self, dim, n_slot, n_out, hidden=256, p=0.2, prior=False, n_zone=1):
        super().__init__()
        self.proj = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, hidden), nn.GELU())
        self.slot_emb = nn.Parameter(torch.randn(n_slot, hidden) * 0.02)
        # fork (idea 3): tokens are (slot, zone) pairs, slot-major. A zone embedding
        # exists only when there is more than one zone, so a 1-zone head has exactly
        # upstream's parameters and state dict.
        self.n_zone = n_zone
        self.zone_emb = nn.Parameter(torch.zeros(n_zone, hidden)) if n_zone > 1 else None'''),
    ('''    def forward(self, x, mask):
        h = self.proj(x) + self.slot_emb
        att = torch.einsum("bsh,oh->bos", h, self.query) / self.hidden ** 0.5
        if self.prior:
            att = att + self.slot_prior.unsqueeze(0)''',
     '''    def forward(self, x, mask):
        Z = self.n_zone
        emb = self.slot_emb if Z == 1 else self.slot_emb.repeat_interleave(Z, 0)
        if self.zone_emb is not None:
            emb = emb + self.zone_emb.repeat(self.slot_emb.shape[0], 1)
        if Z > 1:  # a slot's presence applies to all of its zones
            mask = mask.repeat_interleave(Z, 1)
        h = self.proj(x) + emb
        att = torch.einsum("bsh,oh->bos", h, self.query) / self.hidden ** 0.5
        if self.prior:
            prior_ = self.slot_prior if Z == 1 else self.slot_prior.repeat_interleave(Z, 1)
            att = att + prior_.unsqueeze(0)'''),
    ('''    def __init__(self, backbone, dim, pool="cls_mean", prior=False):''',
     '''    def __init__(self, backbone, dim, pool="cls_mean", prior=False, zones=1):'''),
    ('''        self.head = SlotHead(dim * POOL_PARTS[pool], N_SLOT, len(TARGETS), prior=prior)''',
     '''        self.zones = zones  # fork (idea 3)
        self.head = SlotHead(dim * POOL_PARTS[pool], N_SLOT, len(TARGETS), prior=prior,
                             n_zone=zones)'''),
    ('''        B, S = imgs.shape[:2]
        x = imgs.reshape(B * S, *imgs.shape[2:]).float().div_(255.0)''',
     '''        B, S = imgs.shape[:2]
        # fork (idea 3): (B, S, zones*GROUP, H, W) -> one GROUP-channel image per
        # (slot, zone); zones=1 is upstream's reshape exactly.
        assert imgs.shape[2] % self.zones == 0, (imgs.shape, self.zones)
        x = imgs.reshape(B * S * self.zones, imgs.shape[2] // self.zones,
                         *imgs.shape[3:]).float().div_(255.0)'''),
    ('''        feat = torch.cat(parts, dim=1).reshape(B, S, -1)''',
     '''        feat = torch.cat(parts, dim=1).reshape(B, S * self.zones, -1)'''),
    ('''def build_model(unfreeze_last, source=None, variant="small", pool="cls_mean",
                prior=False):''', '''def build_model(unfreeze_last, source=None, variant="small", pool="cls_mean",
                prior=False, zones=1):'''),
    ('''    return Model(bb, dim, pool=pool, prior=prior)''',
     '''    return Model(bb, dim, pool=pool, prior=prior, zones=zones)'''),
    # fingerprints of a zone model need its full slice count per slot
    ('''def check_fingerprint(model, dev, img_size, expected, tol=FINGERPRINT_TOL, tag=""):
    """Compare against a stored fingerprint; raise when the model is not the same map."""
    got = fingerprint(model, dev, img_size)''',
     '''def check_fingerprint(model, dev, img_size, expected, tol=FINGERPRINT_TOL, tag="",
                      group=None):
    """Compare against a stored fingerprint; raise when the model is not the same map."""
    got = fingerprint(model, dev, img_size, group=group)  # fork: group for zone models'''),
    # training: every zone every step; model built with zones
    ('''                g = int(torch.randint(N_GROUP, (1,)).item())
                imgs = augment(take_group(rows, g))''',
     '''                if ZONES > 1:  # fork (idea 3): every depth zone, every step
                    imgs = augment(rows)
                else:
                    g = int(torch.randint(N_GROUP, (1,)).item())
                    imgs = augment(take_group(rows, g))'''),
    ('''        model = build_model(UNFREEZE_LAST).to(dev)''',
     '''        model = build_model(UNFREEZE_LAST, variant=BACKBONE, zones=ZONES).to(dev)'''),
    # --- idea 4: other backbones through timm (TIMM_BACKBONE above) ---
    # slice band from the FORK cell (wider-band test, 2026-09-30); inference still adopts each
    # member group's own band from its pixel key, so this only matters for cache/train/legacy
    ('''SLICE_BAND = (0.20, 0.80)''',
     '''SLICE_BAND = tuple(FORK.get("band") or (0.20, 0.80))  # fork: upstream (0.20, 0.80)'''),
    ('''UNFREEZE_LAST = 6          # trainable transformer blocks, from the output end''',
     '''UNFREEZE_LAST = int(FORK.get("unfreeze") or 6)  # fork: trainable blocks (upstream 6)'''),
    ('''LR_BACKBONE = 8e-6         # the encoder is adapted, not retrained''',
     '''LR_BACKBONE = float(FORK.get("lr_bb") or 8e-6)  # fork (idea 4): per backbone; upstream 8e-6'''),
    ('''FINGERPRINT_TOL = 2e-3''', TIMM_BACKBONE + '''FINGERPRINT_TOL = 2e-3'''),
    ('''    from transformers import AutoModel
    p = source if source is not None else find_dinov2(variant)''',
     '''    if str(variant).startswith("timm:"):  # fork (idea 4): a timm backbone
        return build_timm_model(variant, pool=pool, prior=prior, zones=zones)
    from transformers import AutoModel
    p = source if source is not None else find_dinov2(variant)'''),
    # fold choice (idea 2)
    ('''    va = np.array([i for i in keep if grp[i] == 0])
    tr = np.array([i for i in keep if grp[i] != 0])''',
     '''    va = np.array([i for i in keep if grp[i] == FOLD])  # fork: fold from the FORK cell
    tr = np.array([i for i in keep if grp[i] != FOLD])'''),
    # save the best model (+ OOF predictions in train mode)
    ('''        if best_state is not None:
            model.load_state_dict(best_state)
''', '''        if best_state is not None:
            model.load_state_dict(best_state)
            save_member(model, dev, cfg, best, best_annot)  # fork: persist the weights
            if FORK["mode"] == "train":
                save_oof(model, dev, cfg, st_tr, Ctr, Mtr, va)
'''),
    # train mode never touches the test set
    ('''        test_preds[cfg["name"]] = predict(model, Cte, Mte, np.arange(len(st_te)), dev,
                                          cfg["img"])''',
     '''        if FORK["mode"] != "train":  # fork: train mode skips the test set
            test_preds[cfg["name"]] = predict(model, Cte, Mte, np.arange(len(st_te)), dev,
                                              cfg["img"])'''),
    ('''    log(f"best on the holdout: {pick} ({results[pick][0]:.4f})")''',
     '''    log(f"best on the holdout: {pick} ({results[pick][0]:.4f})")
    if FORK["mode"] == "train":  # fork: member + OOF saved; no submission to write
        return'''),
    # train mode: the cached pixels replace the header pass + decode of the training set
    ('''    log("header pass: test")''',
     '''    if FORK["mode"] == "train":  # fork (idea 1): pixels from the cache kernels
        st_tr, Ctr, Mtr = load_pixel_cache()
    else:
        st_tr = Ctr = Mtr = None
    if FORK["mode"] == "train":
        hte = htr = None
    else:
      log("header pass: test")'''),
    # predict(): a zone model reads every zone in one pass
    ('''        acc = None
        for g in range(N_GROUP):''',
     '''        acc = None
        if getattr(model, "zones", 1) > 1:  # fork (idea 3): one pass over every zone
            rows = torch.from_numpy(np.ascontiguousarray(cache[sel])).to(dev)
            with torch.autocast("cuda", enabled=dev.type == "cuda"):
                acc = model(rows, m, img_size).float() * N_GROUP  # /N_GROUP below
        for g in (range(N_GROUP) if getattr(model, "zones", 1) == 1 else ()):'''),
    # rehearse mode: point every data path at a stand-in test set of training studies
    ('''ROOT = find_root()
log(f"input root: {ROOT}")''', '''ROOT = find_root()
log(f"input root: {ROOT}")
if FORK["mode"] == "rehearse":  # fork: scoring rehearsal (see make_rehearsal_root)
    ROOT = make_rehearsal_root(ROOT, int(FORK["n_rehearse"]))
    log(f"REHEARSAL root: {ROOT} ({FORK['n_rehearse']} training studies as the test set)")'''),
    # [redacted: detail about the data challenge this code was written for]
    ('''except Exception:
    traceback.print_exc()''', '''except Exception:
    traceback.print_exc()
    if FORK["mode"] in ("rehearse", "cache", "train"):  # fork: surface the error; only
        raise                                            # submission modes fall back to 0.5'''),
    # fork helpers, defined just before main()
    ('''def main():
    write_benchmark_submission()''', '''# ---- fork helpers ----------------------------------------------- #
PACKAGE_DIR = Path("/kaggle/working/package")
PIXEL_DIR = Path("/kaggle/working/pixel_cache")
INFER_CHUNK = 256  # test studies decoded per chunk at inference (~2 GB at 12 slices/slot)


def pixel_key():
    """Everything that decides what a cached / fed slice IS."""
    return {"img": int(IMG), "group": int(GROUP), "slices": int(CACHE_SLICES),
            "crop_mm": float(CROP_MM), "band": [float(x) for x in SLICE_BAND],
            "rules": dict(RULES), "slots": [s[0] for s in SLOTS]}


def save_member(model, dev, cfg, holdout, annot):
    """Write a run's best model in the format infer_from_package() reads, so a later
    submission attaches the package and skips training. The pixel key records exactly
    what the model was trained on (training feeds the cache at its own size)."""
    import json
    assert int(cfg["img"]) == int(IMG), (cfg["img"], IMG)
    PACKAGE_DIR.mkdir(parents=True, exist_ok=True)
    zones = int(getattr(model, "zones", 1))
    tag = FORK.get("member_tag", "")  # fork: distinct ids across backbones/label variants
    mid = f"{cfg['name']}{'_' + tag if tag else ''}_z{zones}_f{FOLD}"
    torch.save({"model": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                "fingerprint": fingerprint(model, dev, int(cfg["img"]),
                                           group=GROUP * zones)},
               PACKAGE_DIR / f"{mid}.pt")
    man_path = PACKAGE_DIR / "manifest.json"
    man = json.loads(man_path.read_text()) if man_path.is_file() else {"members": []}
    man["members"] = [m for m in man["members"] if m["id"] != mid] + [{
        "id": mid, "file": f"{mid}.pt", "fold": int(FOLD), "holdout": float(holdout),
        "annot": None if annot != annot else float(annot),
        "pixel_group": json.dumps(pixel_key(), sort_keys=True),
        "config": {"unfreeze_last": int(UNFREEZE_LAST), "variant": BACKBONE,
                   "pool": "cls_mean", "prior": False, "zones": zones,
                   "epochs": int(EPOCHS), "batch": int(BATCH_STUDIES)}}]
    man_path.write_text(json.dumps(man, indent=2))
    log(f"  fork: saved package member {mid} (holdout {holdout:.4f}) to {PACKAGE_DIR}")


def save_oof(model, dev, cfg, st_tr, Ctr, Mtr, va):
    """Out-of-fold predictions for this fold's holdout (idea 2): pooled over 5 folds,
    every study -- including every gold study -- gets a prediction from a model that
    never trained on it."""
    p = predict(model, Ctr, Mtr, va, dev, int(cfg["img"]))
    df = pd.DataFrame(p, columns=TARGETS)
    df.insert(0, "StudyInstanceUID", [st_tr[i] for i in va])
    df.insert(1, "fold", int(FOLD))
    # a tagged run (another backbone / label variant) gets its own file, so merged
    # packages keep every recipe's OOF instead of one overwriting the other
    tag = FORK.get("member_tag", "")
    df.to_csv(PACKAGE_DIR / f"oof_f{FOLD}{'_' + tag if tag else ''}.csv", index=False)
    log(f"  fork: wrote {len(df)} out-of-fold predictions (fold {FOLD})")


def build_pixel_cache_shard():
    """Idea 1: order + decode the training pixels for shard FORK["shard"] of
    FORK["n_shards"] (studies sorted, every n-th) and save them. No labels, no model."""
    import json
    train_series = pd.read_csv(ROOT / "train_series.csv")
    test_series = pd.read_csv(ROOT / "test_series.csv")
    both = pd.concat([train_series, test_series])
    plane_map = dict(zip(both["SeriesInstanceUID"], both["Anatomical_Plane"]))
    log("header pass: train")
    htr = annotate(walk("train_series"))
    slots = pick_slots(htr, plane_map)
    studies = sorted(slots)
    k, n = int(FORK["shard"]), int(FORK["n_shards"])
    mine = {s: slots[s] for i, s in enumerate(studies) if i % n == k}
    log(f"shard {k}/{n}: {len(mine)} of {len(studies)} studies, {CACHE_SLICES} slices/slot")
    st, C, M = build_cache(mine, plane_map, lat_of(htr, "train "), f"train shard {k}")
    PIXEL_DIR.mkdir(parents=True, exist_ok=True)
    np.save(PIXEL_DIR / f"cache_{k}.npy", C)
    np.save(PIXEL_DIR / f"mask_{k}.npy", M)
    (PIXEL_DIR / f"meta_{k}.json").write_text(json.dumps({
        "shard": k, "n_shards": n, "studies": list(st), "pixel_key": pixel_key(),
        "decode_failed": len(DECODE_FAILED)}))
    log(f"shard {k}: saved cache {C.shape} ({C.nbytes / 1024 ** 3:.1f} GB), "
        f"{len(DECODE_FAILED)} series with decode failures")


class ShardedCache:
    """Several memory-mapped shard arrays behaving like one [study, ...] array for the
    indexing the training code uses (cache[idx] and cache[idx, :, a:b])."""

    def __init__(self, parts):
        self.parts = parts
        self.offsets = np.cumsum([0] + [len(p) for p in parts])
        self.shape = (int(self.offsets[-1]),) + tuple(parts[0].shape[1:])
        self.dtype = parts[0].dtype

    def __len__(self):
        return self.shape[0]

    def __getitem__(self, key):
        idx, rest = (key[0], tuple(key[1:])) if isinstance(key, tuple) else (key, ())
        idx = np.asarray(idx).reshape(-1)
        part = np.searchsorted(self.offsets, idx, side="right") - 1
        first = self.parts[part[0]][(int(idx[0] - self.offsets[part[0]]),) + rest]
        out = np.empty((len(idx),) + first.shape, self.dtype)
        for j, (p, i) in enumerate(zip(part, idx)):
            out[j] = self.parts[p][(int(i - self.offsets[p]),) + rest]
        return out


def load_pixel_cache():
    """Find every cache shard under /kaggle/input, check each was built with exactly this
    pixel recipe, and join them (memory-mapped, so RAM does not bound the cache)."""
    import json
    metas = []
    for root, dirs, files in os.walk("/kaggle/input"):
        if os.path.basename(root) == "competitions":
            dirs[:] = []
            continue
        metas += [Path(root) / f for f in files if f.startswith("meta_") and f.endswith(".json")]
    if not metas:
        raise FileNotFoundError("no pixel cache shards (meta_*.json) under /kaggle/input")
    metas = sorted(metas, key=lambda p: json.loads(p.read_text())["shard"])
    parts, masks, studies = [], [], []
    want = pixel_key()
    for mp in metas:
        meta = json.loads(mp.read_text())
        if meta["pixel_key"] != want:
            raise ValueError(f"{mp}: cache built with {meta['pixel_key']}, this run wants {want}")
        k = meta["shard"]
        parts.append(np.load(mp.parent / f"cache_{k}.npy", mmap_mode="r"))
        masks.append(np.load(mp.parent / f"mask_{k}.npy"))
        studies += meta["studies"]
    n_shards = json.loads(metas[0].read_text())["n_shards"]
    if len(metas) != n_shards:
        raise ValueError(f"found {len(metas)} of {n_shards} cache shards")
    C = ShardedCache(parts)
    log(f"pixel cache: {len(metas)} shards, {C.shape} uint8 (memory-mapped)")
    return studies, C, np.concatenate(masks)


def main():
    # fork: only modes meant for submission write the all-0.5 placeholder. Upstream wrote
    # it in every mode, so our cache/train notebooks carried a submission.csv, appeared in
    # the Submit dialog, and one was submitted by mistake (2026-09-25: "DINOv2 Cache Z4 S2"
    # scored exactly 0.500).
    if FORK["mode"] in ("legacy", "infer", "rehearse"):
        write_benchmark_submission()
    if FORK["mode"] == "cache":  # fork (idea 1)
        build_pixel_cache_shard()
        return'''),
]

# In train mode the test/train header pass + decode block is skipped. Upstream's block
# (from `log("header pass: test")` through the test build_cache line) is re-indented
# under the `else:` the patch above opens, so it only runs in the other modes.
TRAIN_BLOCK_END = '''    st_te, Cte, Mte = build_cache(slots_te, plane_map, lat_of(hte, "test "), "test")'''


REHEARSAL_HELPER = """

def make_rehearsal_root(real, n, seed=0):
    \"\"\"Scoring rehearsal: a stand-in competition folder in /kaggle/working whose test set
    is n random TRAINING studies (links to their DICOM folders), so the inference path
    runs at hidden-test scale with a readable log. Added after the first DINOv2
    submission scored exactly 0.500 -- the signature of the notebook's all-0.5 fallback.\"\"\"
    import os
    import numpy as np
    import pandas as pd
    from pathlib import Path
    fake = Path("/kaggle/working/rehearsal_comp")
    (fake / "test_series").mkdir(parents=True, exist_ok=True)
    for f in ("train.csv", "train_series.csv", "sample_submission.csv"):
        if (real / f).exists() and not (fake / f).exists():
            os.symlink(real / f, fake / f)
    series = pd.read_csv(real / "train_series.csv")
    studies = sorted(series["StudyInstanceUID"].unique())
    pick = sorted(np.random.default_rng(seed).choice(studies, n, replace=False).tolist())
    pd.DataFrame({"StudyInstanceUID": pick}).to_csv(fake / "test.csv", index=False)
    series[series["StudyInstanceUID"].isin(pick)].to_csv(fake / "test_series.csv", index=False)
    for st in pick:
        link = fake / "test_series" / st
        if not link.exists():
            os.symlink(real / "train_series" / st, link)
    return fake
"""

REHEARSAL_CHECK = """# fork: scoring-rehearsal check -- do the predictions actually rank?
if FORK["mode"] == "rehearse":
    import hashlib
    from sklearn.metrics import roc_auc_score
    sub = pd.read_csv("submission.csv")
    n_const = int((sub[TARGETS].nunique() <= 1).sum())
    n_half = int((sub[TARGETS] == 0.5).all(axis=1).sum())
    log(f"REHEARSAL: {len(sub)} rows; constant columns {n_const}/12; all-0.5 rows {n_half}")
    tr = pd.read_csv(ROOT / "train.csv")
    lab = read_labels(tr)
    rep = tr.set_index("StudyInstanceUID")["Report"].fillna("")
    fold = sub["StudyInstanceUID"].map(
        lambda s: int(hashlib.md5(rep.get(s, s).encode()).hexdigest()[:8], 16) % 5)
    hold = sub[fold.values == 0]  # the studies the v1 member never trained on
    y = (lab.loc[hold["StudyInstanceUID"], TARGETS].astype(float) > 0.5).astype(int)
    aucs = {t: roc_auc_score(y[t], hold[t]) for t in TARGETS if y[t].nunique() == 2}
    log(f"REHEARSAL: fold-0 holdout studies {len(hold)}, report-label macro AUC "
        f"{np.mean(list(aucs.values())):.4f} over {len(aucs)} labels "
        f"(training holdout was 0.797; ~0.5 would mean the predictions carry no ranking)")
    for t, a in aucs.items():
        log(f"  {t:<17} {a:.3f}")
"""


# Upstream's infer_from_package() decodes the WHOLE test set into RAM before predicting.
# At 3 slices/slot that is 2 MB per study; at 12 (4 depth zones) it is 8.1 MB, and a large
# [redacted: detail about the data challenge this code was written for]
# scored. The fork replaces the function with a streaming version: same inputs, slot
# choice, laterality and rank-mean output (upstream's tail, verbatim), but each member is
# loaded once and the test set is decoded + predicted INFER_CHUNK studies at a time.
# Upstream's time-budget window surrendering is dropped: at ~1.2 s/study the 9 h limit
# is not the constraint (rehearsal, 2026-09-25).
INFER_START = "def infer_from_package(path, dev):"
INFER_END = "def adopt_config_globals(cfg):"
STREAMING_INFER = """def infer_from_package(path, dev):
    \"\"\"fork: streaming inference from an attached package -- see
    scripts/build_dinov2_fork.py. Members are grouped by the pixels they need (one
    decode per group, as upstream); within a group the test set is processed in chunks
    of INFER_CHUNK studies so RAM never holds more than one chunk's pixels.\"\"\"
    import json
    man = json.loads((Path(path) / "manifest.json").read_text())
    members = man["members"]
    log(f"weights package: {len(members)} member(s) from {path}")

    test_df = pd.read_csv(ROOT / "test.csv")
    test_series = pd.read_csv(ROOT / "test_series.csv")
    plane_map = dict(zip(test_series["SeriesInstanceUID"],
                         test_series["Anatomical_Plane"]))
    hte = annotate(walk("test_series"))
    log(f"test header pass: {len(hte)} series")

    groups = {}
    for m in members:
        groups.setdefault(m["pixel_group"], []).append(m)

    per_member = []
    for gi, (key, gm) in enumerate(groups.items(), 1):
        cfg = json.loads(key)
        adopt_config_globals(cfg)
        slots = pick_slots(hte, plane_map)
        lat = lat_of(hte, "test ")
        studies = sorted(slots)
        log(f"decode group {gi}/{len(groups)}: {cfg['img']}px x {cfg['slices']} slices, "
            f"crop {cfg['crop_mm']} mm -> {len(gm)} member(s), {len(studies)} studies "
            f"in chunks of {INFER_CHUNK}")
        loaded = []
        for m in gm:
            ck = torch.load(Path(path) / m["file"], map_location="cpu", weights_only=False)
            zones = int(m["config"].get("zones", 1))
            model = build_model(int(m["config"]["unfreeze_last"]),
                                variant=m["config"]["variant"],
                                pool=m["config"].get("pool", "cls_mean"),
                                prior=bool(m["config"].get("prior", False)),
                                zones=zones).to(dev)
            model.load_state_dict(ck["model"])
            check_fingerprint(model, dev, IMG, ck["fingerprint"], tag=f"{m['id']}: ",
                              group=GROUP * zones)
            if zones > 1:
                assert CACHE_SLICES == GROUP * zones, (CACHE_SLICES, GROUP, zones)
            loaded.append((m, model, zones))
            del ck
        preds = {m["id"]: [] for m, _, _ in loaded}
        ids = []
        t0 = time.time()
        for c0 in range(0, len(studies), INFER_CHUNK):
            chunk = {st: slots[st] for st in studies[c0:c0 + INFER_CHUNK]}
            st_c, Cc, Mc = build_cache(chunk, plane_map, lat,
                                       f"test g{gi} chunk {c0 // INFER_CHUNK + 1}")
            idx = np.arange(len(st_c))
            for m, model, zones in loaded:
                if zones > 1:  # every zone at once: trained on the whole stack
                    p = predict_member(model, Cc, Mc, idx, dev, IMG, group=CACHE_SLICES,
                                       starts=[0])
                else:
                    p = predict_member(model, Cc, Mc, idx, dev, IMG)
                preds[m["id"]].append(p)
            ids += list(st_c)
            del Cc, Mc
            gc.collect()
            log(f"  group {gi}: {len(ids)}/{len(studies)} studies predicted, "
                f"{time.time() - t0:.0f}s")
        for m, model, zones in loaded:
            per_member.append({"id": m["id"], "ids": ids,
                               "pred": np.concatenate(preds[m["id"]]),
                               "holdout": m.get("holdout")})
        del loaded
        gc.collect()
        if dev.type == "cuda":
            torch.cuda.empty_cache()

"""


def _replace_infer(src):
    """Swap upstream's infer_from_package body (up to its rank-mean tail, kept verbatim)
    for STREAMING_INFER."""
    i = src.index(INFER_START)
    tail = src.index("    # Rank rather than probability", i)
    end = src.index(INFER_END, i)
    assert tail < end, "upstream's rank-mean tail not found inside infer_from_package"
    return src[:i] + STREAMING_INFER + src[tail:end] + src[end:]


def _indent_block(src):
    """Indent upstream's header-pass/decode block (after the patched `else:` line) by
    two spaces so it sits under that else -- a minimal, mechanical change."""
    start = src.index('    else:\n      log("header pass: test")')
    body_start = src.index("\n", src.index('      log("header pass: test")')) + 1
    end = src.index(TRAIN_BLOCK_END) + len(TRAIN_BLOCK_END)
    body = src[body_start:end]
    body = "\n".join(("  " + l) if l.strip() else l for l in body.split("\n"))
    return src[:body_start] + body + src[end:]


def build(fork_cfg):
    nb = json.loads((UPSTREAM / "rsna-knee-baseline-v1.ipynb").read_text())
    code_cells = [c for c in nb["cells"] if c["cell_type"] == "code"]
    for old, new in PATCHES:
        hits = [c for c in code_cells if old in "".join(c["source"])]
        assert len(hits) == 1, f"patch target found in {len(hits)} cells: {old[:70]!r}"
        src = "".join(hits[0]["source"])
        assert src.count(old) == 1, f"patch target not unique: {old[:70]!r}"
        src = src.replace(old, new)
        if old.startswith('    log("header pass: test")'):
            src = _indent_block(src)
        hits[0]["source"] = src.splitlines(keepends=True)
    hits = [c for c in code_cells if INFER_START in "".join(c["source"])]
    assert len(hits) == 1, f"infer_from_package found in {len(hits)} cells"
    hits[0]["source"] = _replace_infer("".join(hits[0]["source"])).splitlines(keepends=True)
    for c in code_cells:  # clear upstream's stored outputs
        c["outputs"], c["execution_count"] = [], None
    config_src = (
        "# fork: run configuration -- see scripts/build_dinov2_fork.py\n"
        f"FORK = {json.dumps(fork_cfg, sort_keys=True)}\n"
        'ZONES = FORK["zones"] if FORK["mode"] in ("cache", "train") else 1\n'
        'FOLD = FORK["fold"]\n'
        'BACKBONE = FORK.get("backbone", "small")  # fork (idea 4)\n'
    ) + (REHEARSAL_HELPER if fork_cfg["mode"] == "rehearse" else "")
    if fork_cfg["mode"] == "rehearse":
        nb["cells"].append({"cell_type": "code", "execution_count": None, "metadata": {},
                            "outputs": [], "source": REHEARSAL_CHECK.splitlines(keepends=True)})
    nb["cells"][0:0] = [
        {"cell_type": "markdown", "metadata": {}, "source": ATTRIBUTION.splitlines(keepends=True)},
        {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
         "source": config_src.splitlines(keepends=True)},
    ]
    # every cell must still be valid Python (IPython magics aside), compiled per cell as
    # Jupyter runs them -- upstream has `from __future__` imports in later cells
    for i, c in enumerate(nb["cells"]):
        if c["cell_type"] == "code":
            src = "".join(c["source"])
            compile("\n".join(l for l in src.splitlines() if not l.lstrip().startswith(("%", "!"))),
                    f"fork_cell_{i}", "exec")
    return nb


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["legacy", "cache", "train", "infer", "rehearse"], default="legacy")
    ap.add_argument("--n-rehearse", type=int, default=1300, help="stand-in test size (rehearse mode)")
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--zones", type=int, default=1)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--n-shards", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch", type=int, default=None, help="studies per step (default 8, 8//zones*2 with zones)")
    ap.add_argument("--package", default=None, help="weights package dataset (infer mode)")
    ap.add_argument("--labels", default=LABELS_DATASET,
                    help="label-table dataset to attach (exactly one report_labels*.csv; idea 5)")
    ap.add_argument("--tag", default="", help="suffix for the kernel id/dir, e.g. 'blend'")
    ap.add_argument("--backbone", default="small",
                    help='DINOv2 size ("small") or "timm:<name>@<px>" (idea 4)')
    ap.add_argument("--lr-bb", type=float, default=None, help="backbone LR (default 8e-6, DINOv2's)")
    ap.add_argument("--grad-ckpt", action="store_true", help="gradient checkpointing (timm, train)")
    ap.add_argument("--band", type=float, nargs=2, default=None, metavar=("LO", "HI"),
                    help="slice band of the ordered stack (upstream 0.20 0.80); cache + train")
    ap.add_argument("--unfreeze", type=int, default=None,
                    help="DINOv2 blocks trained, from the output end (upstream 6 of 12)")
    ap.add_argument("--cache-tag", default="",
                    help="train mode: read the cache kernels built with this --tag (e.g. b10)")
    ap.add_argument("--machine", default="NvidiaTeslaT4",
                    help="GPU machine_shape (e.g. NvidiaTeslaP100 to test what a scoring run might get)")
    args = ap.parse_args()
    if args.mode in ("infer", "rehearse") and not args.package:
        ap.error(f"--mode {args.mode} needs --package")
    batch = args.batch or (8 if args.zones == 1 else max(2, 8 // args.zones * 2))
    fork_cfg = {"mode": args.mode, "fold": args.fold, "zones": args.zones, "shard": args.shard,
                "n_shards": args.n_shards, "epochs": args.epochs, "batch": batch,
                "n_rehearse": args.n_rehearse}
    # idea 4 keys only when used, so earlier DINOv2 configs are unchanged
    if args.backbone != "small":
        fork_cfg["backbone"] = args.backbone
    if args.lr_bb is not None:
        fork_cfg["lr_bb"] = args.lr_bb
    if args.grad_ckpt:
        fork_cfg["grad_ckpt"] = True
    if args.band is not None:
        fork_cfg["band"] = list(args.band)
    if args.unfreeze is not None:
        fork_cfg["unfreeze"] = args.unfreeze
    if args.tag and args.mode == "train":
        fork_cfg["member_tag"] = args.tag
    nb = build(fork_cfg)

    upstream_meta = json.loads((UPSTREAM / "original-kernel-metadata.json").read_text())
    tag = {"legacy": "fork", "cache": f"cache-z{args.zones}-s{args.shard}",
           "train": f"train-z{args.zones}-f{args.fold}", "infer": "infer",
           "rehearse": "rehearse"}[args.mode]
    if args.tag:
        tag += f"-{args.tag}"
    out = ROOT / "kaggle_kernels" / f"dinov2_{tag.replace('-', '_')}"
    ctag = f"-{args.cache_tag}" if args.cache_tag else ""
    cache_kernels = [f"{OWNER}/rsna-knee-dinov2-cache-z{args.zones}-s{i}{ctag}" for i in range(args.n_shards)]
    meta = {
        "id": f"{OWNER}/rsna-knee-dinov2-{tag}",
        "title": f"RSNA Knee - DINOv2 {tag.replace('-', ' ').title()}",
        "code_file": "dinov2_fork.ipynb",
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": args.mode != "cache",  # the cache build is CPU work: no GPU quota
        # timm weights are downloaded at training time; scoring notebooks stay offline
        "enable_internet": args.mode == "train" and args.backbone.startswith("timm:"),
        "docker_image": upstream_meta["docker_image"],  # upstream's pinned environment
        "dataset_sources": [args.labels] + ([args.package] if args.package else []),
        "model_sources": upstream_meta["model_sources"],
        "competition_sources": ["rsna-knee-abnormality-detection"],
        "kernel_sources": cache_kernels if args.mode == "train" else [],
    }
    if args.mode != "cache":
        meta["machine_shape"] = args.machine
    out.mkdir(parents=True, exist_ok=True)
    (out / "dinov2_fork.ipynb").write_text(json.dumps(nb, indent=1))
    (out / "kernel-metadata.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(f"wrote {out.relative_to(ROOT)} ({meta['id']}, FORK={fork_cfg})")


if __name__ == "__main__":
    main()
