"""Tests for src/models/efficiency.py, with hand-computed MAC counts."""

import sys
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from models.backbone3d import Simple3DCNN
from models.efficiency import count_macs, measure_latency
from models.knee_model import KneeMultiViewModel


def test_conv3d_macs_hand_computed():
    # Conv3d(1, 2, kernel_size=3, padding=1) on (1,1,4,4,4) -> output (1,2,4,4,4)
    # (padding=1, stride=1, kernel=3 keeps spatial size unchanged).
    # MACs = batch * out_channels * spatial * (in_channels/groups * kD*kH*kW)
    #      = 1 * 2 * (4*4*4) * (1 * 3*3*3) = 1*2*64*27 = 3456
    conv = nn.Conv3d(1, 2, kernel_size=3, padding=1)
    x = torch.randn(1, 1, 4, 4, 4)
    macs = count_macs(conv, lambda: conv(x))
    assert macs == 3456


def test_conv3d_macs_scales_with_batch_and_out_channels():
    conv = nn.Conv3d(2, 4, kernel_size=3, padding=1)
    x = torch.randn(3, 2, 4, 4, 4)
    # MACs = 3 * 4 * 64 * (2 * 27) = 3*4*64*54 = 41472
    macs = count_macs(conv, lambda: conv(x))
    assert macs == 41472


def test_linear_macs_hand_computed():
    # Linear(10, 5) on (3, 10) -> output (3, 5). MACs = out.numel() * in_features = 15*10 = 150
    lin = nn.Linear(10, 5)
    x = torch.randn(3, 10)
    macs = count_macs(lin, lambda: lin(x))
    assert macs == 150


def test_combined_model_sums_conv_and_linear_macs():
    class Tiny(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv3d(1, 2, kernel_size=3, padding=1)
            self.pool = nn.AdaptiveAvgPool3d(1)
            self.fc = nn.Linear(2, 5)

        def forward(self, x):
            x = self.conv(x)
            x = torch.flatten(self.pool(x), 1)
            return self.fc(x)

    model = Tiny()
    x = torch.randn(1, 1, 4, 4, 4)
    macs = count_macs(model, lambda: model(x))
    # conv: 1*2*64*27 = 3456 (as above). fc: out.numel()=1*5=5, in_features=2 -> 10.
    assert macs == 3456 + 10


def test_count_macs_restores_training_mode():
    model = nn.Linear(4, 4)
    model.train()
    count_macs(model, lambda: model(torch.randn(1, 4)))
    assert model.training is True
    model.eval()
    count_macs(model, lambda: model(torch.randn(1, 4)))
    assert model.training is False


def test_count_macs_on_real_model_is_positive_and_sane():
    backbone = Simple3DCNN(in_channels=1, stem_channels=2, channels=(4, 8))
    model = KneeMultiViewModel(n_classes=12, backbone=backbone)
    image = torch.randn(1, 3, 16, 16, 16)
    plane_present = torch.ones(1, 3, dtype=torch.bool)
    macs = count_macs(model, lambda: model(image, plane_present))
    assert macs > 0


def test_measure_latency_returns_expected_keys_and_positive_time():
    model = nn.Linear(16, 16)
    x = torch.randn(4, 16)
    result = measure_latency(lambda: model(x), n_warmup=1, n_repeats=3)
    assert set(result.keys()) == {"mean_ms", "std_ms", "min_ms", "max_ms", "n_repeats"}
    assert result["mean_ms"] > 0
    assert result["n_repeats"] == 3
    assert result["min_ms"] <= result["mean_ms"] <= result["max_ms"]
