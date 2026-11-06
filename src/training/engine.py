"""Training loop primitives (module 8). Deliberately small and reusable --
the "assemble everything from real files on disk" wiring (load the manifest,
the training label table, the fold assignment, build DataLoaders) lives in
scripts/train.py, not here, so these functions stay unit-testable against
synthetic data without touching any of that (same split this project already
uses elsewhere, e.g. final_labels.py's functions vs. build_training_labels.py's
script).
"""

import random
import subprocess
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]


def set_seed(seed: int):
    """Seeds Python's random, numpy, and torch (CPU + CUDA + MPS where
    applicable) -- [redacted: private project notes]'s "fix seeds, log configs" working agreement.
    Does not make CUDA convolutions bit-exact (cudnn.deterministic is a
    separate, throughput-costly flag not set here); this seeds initialization
    and data shuffling, not every possible source of nondeterminism.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def git_commit_hash() -> str:
    """Current commit hash, or "unknown" (never raises) -- for the "config +
    commit hash" reproducibility record, not something a training run should
    fail over if git happens to be unavailable.
    """
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:
        return "unknown"


def train_one_epoch(model, loader, optimizer, loss_fn, device):
    """One pass over `loader`, updating `model`'s weights. Returns the mean
    per-batch loss (each batch's loss is itself already a mean over its own
    unmasked cells -- see MaskedBCEWithLogitsLoss -- so this is a mean of
    means, fine for a progress readout, not claimed to be the exact
    dataset-wide per-cell mean loss).
    """
    model.train()
    total_loss = 0.0
    n_batches = 0
    for batch in loader:
        image = batch["image"].to(device)
        plane_present = batch["plane_present"].to(device)
        label = batch["label"].to(device)
        label_mask = batch["label_mask"].to(device)
        weight = batch.get("label_weight")
        weight = weight.to(device) if weight is not None else None

        optimizer.zero_grad()
        logits = model(image, plane_present)
        loss = loss_fn(logits, label, label_mask, weight)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1
    return total_loss / max(n_batches, 1)


@torch.no_grad()
def predict(model, loader, device):
    """Runs `model` over every item in `loader` (no gradient, eval mode).
    Returns (study_ids: list[str], y_true, y_pred, label_mask) as numpy
    arrays aligned by row -- y_pred is sigmoid probabilities, ready for
    training.metrics.masked_macro_auc or kaggle_macro_auc.
    """
    model.eval()
    study_ids = []
    all_labels, all_probs, all_masks = [], [], []
    for batch in loader:
        image = batch["image"].to(device)
        plane_present = batch["plane_present"].to(device)
        logits = model(image, plane_present)
        probs = torch.sigmoid(logits)

        study_ids.extend(batch["study_id"])
        all_labels.append(batch["label"].numpy())
        all_probs.append(probs.cpu().numpy())
        all_masks.append(batch["label_mask"].numpy())

    return (
        study_ids,
        np.concatenate(all_labels, axis=0),
        np.concatenate(all_probs, axis=0),
        np.concatenate(all_masks, axis=0),
    )
