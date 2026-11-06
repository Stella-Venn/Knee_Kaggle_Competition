"""Real training entry point: wires the manifest, the training label table,
and the fold assignment (all on disk) into KneeStudyDataset/DataLoader, then
runs training.engine's loop.

Needs the volume cache populated for the studies it will actually train on
(scripts/precompute_volume_cache.py) -- today that's only the 1 downloaded
test study, so this script runs correctly but isn't a real training run yet
[redacted: note about institution-specific computing resources]
exists now so the wiring is in place and tested end-to-end via the synthetic
smoke test (tests/test_engine.py); running it for real is the next step once
the cache exists at scale.

Usage: python scripts/train.py [--config path/to/config.json] [--out-dir data/runs/<name>]
       [--limit-studies N]   # only the first N studies with a cached volume -- for a
                              # real (not synthetic) local smoke run against whatever's cached
"""

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from data.dataset import KneeStudyDataset, select_canonical_series
from data.sampler import compute_pos_weights, fold_split
from models.knee_model import KneeMultiViewModel
from training.config import TrainConfig, load_config, save_config
from training.engine import git_commit_hash, predict, set_seed, train_one_epoch
from training.loss import MaskedBCEWithLogitsLoss
from training.metrics import masked_macro_auc
from utils.constants import LABEL_COLUMNS


def load_series_by_study(manifest_path):
    by_study = defaultdict(list)
    with open(manifest_path, newline="") as f:
        for row in csv.DictReader(f):
            if row["split"] == "train_series":
                by_study[row["StudyInstanceUID"]].append(row)
    return by_study


def studies_with_cache(study_ids, series_by_study, cache_dir, planes=("Sagittal", "Coronal", "Axial")):
    """Filters to study_ids that have at least one canonical-plane series
    actually present in `cache_dir` -- most studies won't, until the cache is
    populated at scale (see module docstring).
    """
    cache_dir = Path(cache_dir)
    keep = []
    for sid in study_ids:
        canonical = select_canonical_series(series_by_study.get(sid, []), planes)
        if any((cache_dir / sid / f"{series_id}.npy").exists() for series_id in canonical.values()):
            keep.append(sid)
    return keep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--out-dir", default=str(ROOT / "data" / "runs" / "default"))
    ap.add_argument("--limit-studies", type=int, default=None)
    args = ap.parse_args()

    config = load_config(args.config) if args.config else TrainConfig()
    set_seed(config.seed)

    labels_npz = np.load(ROOT / config.labels_path, allow_pickle=True)
    study_ids_all = list(labels_npz["study_ids"])
    labels_all = labels_npz["labels"]

    series_by_study = load_series_by_study(ROOT / config.manifest_path)
    cache_dir = Path(config.cache_dir) / "x".join(f"{v:g}" for v in config.target_spacing)
    available = studies_with_cache(study_ids_all, series_by_study, cache_dir)
    if args.limit_studies:
        available = available[: args.limit_studies]
    print(f"{len(available)}/{len(study_ids_all)} studies have a cached volume at {cache_dir}")
    if len(available) < 4:
        raise SystemExit(
            f"Only {len(available)} studies cached -- not enough to train/validate. "
            "Run scripts/precompute_volume_cache.py first (see its docstring for the "
            "local-vs-full-dataset caveat)."
        )

    idx_by_id = {sid: i for i, sid in enumerate(study_ids_all)}
    keep_idx = [idx_by_id[sid] for sid in available]
    labels = labels_all[keep_idx]
    weights = labels_npz["weights"][keep_idx] if "weights" in labels_npz else None

    rng = np.random.default_rng(config.seed)
    fold = (rng.permutation(len(available)) % 2)  # simple 50/50 split for a real-but-tiny local run
    train_idx, val_idx = fold_split(available, fold, val_fold=0)
    print(f"train={len(train_idx)} val={len(val_idx)}")

    full_ds = KneeStudyDataset(available, labels, series_by_study, cache_dir=cache_dir,
                                target_shape=config.target_shape, weights=weights)
    train_loader = torch.utils.data.DataLoader(
        torch.utils.data.Subset(full_ds, train_idx), batch_size=config.batch_size, shuffle=True,
    )
    val_loader = torch.utils.data.DataLoader(
        torch.utils.data.Subset(full_ds, val_idx), batch_size=config.batch_size, shuffle=False,
    )

    model = KneeMultiViewModel(
        n_classes=len(LABEL_COLUMNS), hidden_dim=config.hidden_dim, channels=config.backbone_channels,
    ).to(config.device)
    pos_weight = compute_pos_weights(
        labels[train_idx], max_weight=config.max_pos_weight,
        cell_weights=None if weights is None else weights[train_idx],
    )
    loss_fn = MaskedBCEWithLogitsLoss(
        pos_weight=pos_weight, nan_pos_weight_fallback=config.nan_pos_weight_fallback,
    ).to(config.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)

    # Best-checkpoint tracking, not just the final epoch: a run can (and, on the
    # pilot data, does -- see [redacted: private project notes], 2026-09-22) overfit well before
    # n_epochs is reached, at which point the FINAL checkpoint is the worst
    # generalizing one in the whole run, not the best.
    #
    # Saved every epoch, not just once at the end (2026-09-23), matching the
    # Kaggle training kernel's own fix: a long run should survive an
    # interruption with whatever progress it made intact, not lose everything.
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    def _save_progress(best_state, best_epoch, best_val_macro, history):
        torch.save(model.state_dict(), out_dir / "model_final.pt")
        if best_state is not None:
            torch.save(best_state, out_dir / "model_best.pt")
        save_config(config, out_dir / "config.json", extra={
            "git_commit": git_commit_hash(), "history": history, "n_studies_used": len(available),
            "best_epoch": best_epoch, "best_val_macro_auc": best_val_macro if best_epoch else None,
        })

    history = []
    best_val_macro, best_epoch, best_state = float("-inf"), None, None
    for epoch in range(config.n_epochs):
        train_loss = train_one_epoch(model, train_loader, optimizer, loss_fn, config.device)
        _, y_true, y_pred, mask = predict(model, val_loader, config.device)
        val_macro, per_col, degenerate = masked_macro_auc(y_true, y_pred, mask, LABEL_COLUMNS)
        print(f"epoch {epoch+1}/{config.n_epochs}  train_loss={train_loss:.4f}  "
              f"val_macro_auc={val_macro:.4f}  degenerate={len(degenerate)}/{len(LABEL_COLUMNS)}")
        history.append({"epoch": epoch + 1, "train_loss": train_loss, "val_macro_auc": val_macro})
        if val_macro == val_macro and val_macro > best_val_macro:  # val_macro==val_macro excludes NaN
            best_val_macro, best_epoch = val_macro, epoch + 1
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        _save_progress(best_state, best_epoch, best_val_macro, history)

    print(f"Best epoch: {best_epoch} (val_macro_auc={best_val_macro:.4f})" if best_epoch
          else "No epoch had a defined val_macro_auc (every epoch fully degenerate)")
    print(f"Saved checkpoints and config to {out_dir}")


if __name__ == "__main__":
    main()
