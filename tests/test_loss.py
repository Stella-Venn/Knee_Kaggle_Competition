"""Tests for src/training/loss.py, with hand-computed BCE values."""

import math
import sys
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from training.loss import MaskedBCEWithLogitsLoss


def test_matches_plain_bce_when_nothing_masked_and_no_pos_weight():
    logits = torch.tensor([[0.5, -1.0], [2.0, 0.1]])
    targets = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    mask = torch.ones_like(targets, dtype=torch.bool)
    loss_fn = MaskedBCEWithLogitsLoss()
    got = loss_fn(logits, targets, mask)
    expected = F.binary_cross_entropy_with_logits(logits, targets, reduction="mean")
    assert got.item() == pytest.approx(expected.item())


def test_masked_cells_are_fully_excluded_regardless_of_their_target_value():
    logits = torch.tensor([[0.5, -1.0, 3.0]])
    targets_a = torch.tensor([[1.0, 0.0, 0.0]])
    targets_b = torch.tensor([[1.0, 0.0, 999.0]])  # garbage in the masked-out cell
    mask = torch.tensor([[True, True, False]])
    loss_fn = MaskedBCEWithLogitsLoss()
    loss_a = loss_fn(logits, targets_a, mask)
    loss_b = loss_fn(logits, targets_b, mask)
    assert loss_a.item() == pytest.approx(loss_b.item())
    # and it should match the 2-element mean computed by hand
    expected = F.binary_cross_entropy_with_logits(logits[:, :2], targets_a[:, :2], reduction="mean")
    assert loss_a.item() == pytest.approx(expected.item())


def test_hand_computed_single_element():
    # BCE-with-logits(logit=0, target=1) = -log(sigmoid(0)) = -log(0.5) = ln(2)
    logits = torch.tensor([[0.0]])
    targets = torch.tensor([[1.0]])
    mask = torch.tensor([[True]])
    loss_fn = MaskedBCEWithLogitsLoss()
    got = loss_fn(logits, targets, mask).item()
    assert got == pytest.approx(math.log(2), abs=1e-6)


def test_pos_weight_scales_positive_term():
    # single positive example, pos_weight=3: loss = 3 * -log(sigmoid(logit))
    logits = torch.tensor([[0.0]])
    targets = torch.tensor([[1.0]])
    mask = torch.tensor([[True]])
    loss_fn = MaskedBCEWithLogitsLoss(pos_weight=[3.0])
    got = loss_fn(logits, targets, mask).item()
    assert got == pytest.approx(3 * math.log(2), abs=1e-6)


def test_nan_pos_weight_is_replaced_not_propagated():
    logits = torch.tensor([[0.0, 0.0]])
    targets = torch.tensor([[1.0, 1.0]])
    mask = torch.tensor([[True, True]])
    loss_fn = MaskedBCEWithLogitsLoss(pos_weight=[float("nan"), 3.0], nan_pos_weight_fallback=1.0)
    got = loss_fn(logits, targets, mask).item()
    assert not math.isnan(got)
    # column 0 falls back to weight 1.0, column 1 uses 3.0 -> mean of (log2, 3*log2)
    expected = (math.log(2) + 3 * math.log(2)) / 2
    assert got == pytest.approx(expected, abs=1e-6)


def test_nan_pos_weight_column_fully_masked_out_still_finite():
    # the exact failure mode this class exists to prevent: a NaN-pos_weight column
    # that's ALSO entirely masked out in this batch must not poison the loss via 0*NaN.
    logits = torch.tensor([[1.0, 0.0]])
    targets = torch.tensor([[123.0, 1.0]])  # garbage in the NaN-weighted, masked-out cell
    mask = torch.tensor([[False, True]])
    loss_fn = MaskedBCEWithLogitsLoss(pos_weight=[float("nan"), 1.0])
    got = loss_fn(logits, targets, mask).item()
    assert not math.isnan(got)
    assert got == pytest.approx(math.log(2), abs=1e-6)


def test_all_masked_batch_returns_zero_not_nan():
    logits = torch.tensor([[0.5, -1.0]])
    targets = torch.tensor([[1.0, 0.0]])
    mask = torch.zeros_like(targets, dtype=torch.bool)
    loss_fn = MaskedBCEWithLogitsLoss()
    got = loss_fn(logits, targets, mask)
    assert got.item() == 0.0


def test_pos_weight_buffer_moves_with_to_calls():
    # Regression test for a real bug (found on an actual Kaggle GPU run,
    # 2026-09-22): pos_weight is a registered buffer, but the loss module
    # itself was never given its own .to(device) call anywhere this project
    # constructs it (only the model was moved) -- invisible on CPU-only local
    # testing, since "cpu" is already where everything lives, but a hard
    # RuntimeError on CUDA ("Expected all tensors to be on the same device").
    # Can't exercise an actual CUDA device here (no GPU in this environment),
    # but dtype is the same buffer-movement mechanism .to(device) uses, so
    # this catches a regression where pos_weight stops being a real buffer
    # (e.g. reverts to a plain non-buffer attribute untouched by .to()).
    loss_fn = MaskedBCEWithLogitsLoss(pos_weight=[1.0, 2.0, float("nan")])
    assert loss_fn.pos_weight.dtype == torch.float32
    loss_fn = loss_fn.to(dtype=torch.float64)
    assert loss_fn.pos_weight.dtype == torch.float64


def test_gradients_flow_and_are_finite():
    logits = torch.tensor([[0.5, -1.0, 2.0]], requires_grad=True)
    targets = torch.tensor([[1.0, 0.0, 1.0]])
    mask = torch.tensor([[True, True, False]])
    loss_fn = MaskedBCEWithLogitsLoss(pos_weight=[2.0, float("nan"), 1.0])
    loss = loss_fn(logits, targets, mask)
    loss.backward()
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()
    assert logits.grad[0, 2].item() == 0.0  # masked-out cell gets zero gradient


def test_per_cell_weight_is_a_weighted_mean_and_none_is_unchanged():
    import torch
    loss_fn = MaskedBCEWithLogitsLoss()
    logits = torch.tensor([[0.0, 2.0], [1.0, -1.0]])
    targets = torch.tensor([[1.0, 0.0], [0.0, 0.0]])
    mask = torch.tensor([[True, True], [True, False]])
    base = loss_fn(logits, targets, mask)
    assert torch.allclose(base, loss_fn(logits, targets, mask, torch.ones(2, 2)))
    # weight 0 on a cell == masking it out
    w = torch.tensor([[1.0, 0.0], [1.0, 1.0]])
    masked = loss_fn(logits, targets, torch.tensor([[True, False], [True, False]]))
    assert torch.allclose(loss_fn(logits, targets, mask, w), masked)
    # hand-computed weighted mean: cells (0,0)=log(2) w=1, (0,1)=softplus(2) w=0.5, (1,0)=softplus(1) w=1
    import math
    w = torch.tensor([[1.0, 0.5], [1.0, 1.0]])
    sp = lambda x: math.log1p(math.exp(x))
    expected = (math.log(2) + 0.5 * sp(2.0) + sp(1.0)) / 2.5
    assert abs(loss_fn(logits, targets, mask, w).item() - expected) < 1e-6
