"""Tests for src/models/backbone3d.py."""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from models.backbone3d import ConvBlock3D, Simple3DCNN, count_parameters


def test_simple3dcnn_output_shape_matches_feature_dim():
    model = Simple3DCNN(in_channels=1, stem_channels=2, channels=(4, 8))
    x = torch.randn(3, 1, 16, 16, 16)
    out = model(x)
    assert out.shape == (3, model.feature_dim)
    assert model.feature_dim == 8


def test_simple3dcnn_handles_small_spatial_input():
    # tiny input should survive repeated stride-2 downsampling without erroring
    model = Simple3DCNN(in_channels=1, stem_channels=2, channels=(4,))
    out = model(torch.randn(1, 1, 8, 8, 8))
    assert out.shape == (1, 4)


def test_simple3dcnn_multichannel_input():
    model = Simple3DCNN(in_channels=3, stem_channels=2, channels=(4,))
    out = model(torch.randn(2, 3, 8, 8, 8))
    assert out.shape == (2, 4)


def test_count_parameters_matches_manual_sum():
    model = Simple3DCNN(in_channels=1, stem_channels=2, channels=(4,))
    manual = sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert count_parameters(model) == manual
    assert count_parameters(model) > 0


def test_count_parameters_trainable_only_excludes_frozen():
    model = Simple3DCNN(in_channels=1, stem_channels=2, channels=(4,))
    total_before = count_parameters(model, trainable_only=False)
    for p in model.stem.parameters():
        p.requires_grad = False
    trainable_after = count_parameters(model, trainable_only=True)
    assert trainable_after < total_before


def test_convblock3d_downsamples_spatially():
    block = ConvBlock3D(1, 4, kernel_size=3, stride=2)
    out = block(torch.randn(1, 1, 16, 16, 16))
    assert out.shape == (1, 4, 8, 8, 8)
