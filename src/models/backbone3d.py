"""A small 3D CNN backbone: one-channel (grayscale) volume in, a feature
vector out. Deliberately small (config-driven channel list, default 4 stages
16->32->64->128) rather than reaching for a large pretrained-style 3D
architecture on day one -- [redacted: private project notes]'s engineering priorities put "get the
validation split and data quality right" ahead of architecture novelty, and
there's a separate efficiency track to budget for (see `models/efficiency.py`
for FLOPs/param/latency tracking). This is the inference-safe, image-only
baseline the gameplan asks for; a text/contrastive branch is deferred (see
[redacted: private project notes], "Report text" decision log), not built here.

Kept independent of anything plane/study-specific -- `KneeMultiViewModel`
(knee_model.py) decides how a study's multiple planes feed this backbone.
"""

import torch
import torch.nn as nn


class ConvBlock3D(nn.Module):
    """Conv3d -> BatchNorm3d -> ReLU, stride-2 spatial downsampling."""

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=2):
        super().__init__()
        self.conv = nn.Conv3d(
            in_channels, out_channels, kernel_size=kernel_size,
            stride=stride, padding=kernel_size // 2, bias=False,
        )
        self.bn = nn.BatchNorm3d(out_channels)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class Simple3DCNN(nn.Module):
    """(B, 1, D, H, W) -> (B, feature_dim). `channels` is the output channel
    count of each downsampling stage after the stem, e.g. (16, 32, 64, 128)
    (the default) downsamples spatially by 2 five times total (stem + 4
    stages) -- a (32, 256, 256) input reaches (1, 8, 8) before the final
    pool, small enough to keep parameter count and compute low. A shorter
    `channels` tuple (e.g. (4, 8) in tests) makes a much smaller network for
    fast smoke tests, per [redacted: private project notes]'s "fast smoke-test pipeline" priority.
    """

    def __init__(self, in_channels=1, stem_channels=8, channels=(16, 32, 64, 128)):
        super().__init__()
        self.stem = ConvBlock3D(in_channels, stem_channels, kernel_size=7, stride=2)
        stages = []
        prev = stem_channels
        for c in channels:
            stages.append(ConvBlock3D(prev, c, kernel_size=3, stride=2))
            prev = c
        self.stages = nn.Sequential(*stages)
        self.pool = nn.AdaptiveAvgPool3d(1)
        self.feature_dim = prev

    def forward(self, x):
        x = self.stem(x)
        x = self.stages(x)
        x = self.pool(x)
        return torch.flatten(x, 1)


def count_parameters(model, trainable_only=True):
    """Total parameter count, for the efficiency track and for sanity-checking
    that a "small" backbone actually is one.
    """
    params = model.parameters()
    if trainable_only:
        params = (p for p in params if p.requires_grad)
    return sum(p.numel() for p in params)
