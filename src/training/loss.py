"""Masked, per-column-weighted multi-label loss for the training loop.

[redacted: note about the author's personal workflow or decisions]
needed by the training label table (module 6's `final_labels.py`):

1. Per-cell masking -- most columns are MASKED (no label) for most studies
   (e.g. ACL is labeled for only ~2910/4407 studies today). A masked-out
   cell's `target` value is never a real 0/1, so it must never contribute to
   the loss, not even a small pull toward 0.
2. `sampler.compute_pos_weights` can return NaN for a column with zero known
   positives (documented there as visible-by-design, not silently
   defaulted). Passing that straight to `F.binary_cross_entropy_with_logits`
   would produce NaN for every element of that column -- including masked-out
   ones, since 0 * NaN = NaN in IEEE floats, not 0. Masking AFTER computing
   per-element loss would therefore silently poison the whole batch's loss
   with NaN. `nan_pos_weight_fallback` replaces NaN weights before the loss
   computation, not after, so this can't happen.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MaskedBCEWithLogitsLoss(nn.Module):
    def __init__(self, pos_weight=None, nan_pos_weight_fallback=1.0):
        """pos_weight: (n_classes,) array-like, may contain NaN (see module
        docstring) -- e.g. sampler.compute_pos_weights's direct output.
        nan_pos_weight_fallback: weight substituted for any NaN entry.
        """
        super().__init__()
        if pos_weight is not None:
            pos_weight = torch.as_tensor(pos_weight, dtype=torch.float32).clone()
            pos_weight[torch.isnan(pos_weight)] = nan_pos_weight_fallback
            self.register_buffer("pos_weight", pos_weight)
        else:
            self.pos_weight = None

    def forward(self, logits, targets, mask, weight=None):
        """logits, targets, mask: all (batch, n_classes). `targets` at a
        masked-out (mask=False) cell may hold any value -- it's excluded
        before it can affect the result, not clamped or relied on to be 0.
        weight: optional (batch, n_classes) per-cell weight (e.g. down-weighted
        PU negatives for Synovitis, 2026-09-24); None means all 1.0. The mean
        is weighted: sum(loss * mask * weight) / sum(mask * weight).
        Returns a scalar: mean loss over unmasked cells only, or exactly
        0.0 (not NaN) if a batch happens to have zero unmasked cells at all.
        """
        mask = mask.to(dtype=logits.dtype)
        if weight is not None:
            mask = mask * weight.to(dtype=logits.dtype)
        per_element = F.binary_cross_entropy_with_logits(
            logits, targets.to(dtype=logits.dtype), pos_weight=self.pos_weight, reduction="none",
        )
        n_unmasked = mask.sum()
        if n_unmasked == 0:
            return torch.zeros((), dtype=logits.dtype, device=logits.device)
        return (per_element * mask).sum() / n_unmasked
