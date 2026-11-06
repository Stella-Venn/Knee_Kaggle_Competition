"""Image-only multi-view model: wraps a per-plane 3D backbone with masked
pooling across a study's canonical planes, per gameplan module 7 ("image
backbone first, image-only... text/contrastive branch deferred").

Design decision worth stating explicitly: `KneeStudyDataset` (module 6)
returns one tensor of shape (n_planes, D, H, W) per study, stacking
Sagittal/Coronal/Axial as if they were channels. They are NOT channels in
the usual sense (RGB-style, pixel-aligned) -- they're three separate
acquisitions of different anatomical planes, not co-registered voxel-for-
voxel. Feeding that tensor into a single 3D conv treating planes as input
channels would silently assume a spatial correspondence between them that
doesn't exist. Instead, `KneeMultiViewModel` runs each plane through the
SAME shared-weight 3D backbone independently (folding planes into the batch
dimension), then combines the resulting per-plane feature vectors -- the
standard "multi-view CNN" pattern for exactly this kind of data (distinct
non-co-registered series of one study), not a novel choice. No change to
`dataset.py` was needed for this -- it already returns per-plane volumes
separately; this module is what treats them as views rather than channels.

Masked pooling detail: the backbone is run ONLY on planes flagged present in
`plane_present`, not on the zero-filled placeholder for a missing plane and
then masked out afterward -- running the backbone on zero-filled volumes
would let their all-zero activations pollute BatchNorm's batch statistics
during training even though their contribution to the pooled output is
masked to zero either way. In the current training data this distinction is
moot (the full manifest was checked, 2026-09-22: all 4407 training studies
have all 3 canonical planes present -- see [redacted: private project notes]), but it's real
[redacted: detail about the data challenge this code was written for]
"""

import torch
import torch.nn as nn

from models.backbone3d import Simple3DCNN


class KneeMultiViewModel(nn.Module):
    def __init__(self, n_classes=12, hidden_dim=None, backbone=None, **backbone_kwargs):
        super().__init__()
        self.backbone = backbone if backbone is not None else Simple3DCNN(**backbone_kwargs)
        feat_dim = self.backbone.feature_dim
        layers = []
        if hidden_dim:
            layers += [nn.Linear(feat_dim, hidden_dim), nn.ReLU(inplace=True)]
            feat_dim = hidden_dim
        layers.append(nn.Linear(feat_dim, n_classes))
        self.head = nn.Sequential(*layers)

    def forward(self, image, plane_present):
        """image: (B, P, D, H, W) float. plane_present: (B, P) bool -- which
        of the P planes are real data vs. a missing-plane zero placeholder
        (see KneeStudyDataset). Returns (B, n_classes) logits (unbounded,
        for BCEWithLogitsLoss -- no sigmoid here).
        """
        B, P, D, H, W = image.shape
        if plane_present.shape != (B, P):
            raise ValueError(f"plane_present shape {tuple(plane_present.shape)} != {(B, P)}")

        flat_image = image.reshape(B * P, 1, D, H, W)
        flat_present = plane_present.reshape(B * P)
        feat_dim = self.backbone.feature_dim

        if flat_present.any():
            present_feats = self.backbone(flat_image[flat_present])
            # Allocate in the BACKBONE's output dtype, not the input's: under CUDA
            # autocast the backbone returns float16 while the input is float32,
            # and index_put across dtypes crashed the first full GPU run
            # (2026-09-25). new_zeros is a fresh tensor, so the in-place write
            # doesn't touch anything autograd needs.
            feats = present_feats.new_zeros(B * P, feat_dim)
            feats[flat_present] = present_feats
        else:
            feats = torch.zeros(B * P, feat_dim, device=image.device, dtype=image.dtype)
        feats = feats.view(B, P, feat_dim)

        mask = plane_present.unsqueeze(-1).to(feats.dtype)  # (B, P, 1)
        counts = mask.sum(dim=1).clamp(min=1.0)  # avoid /0 if a study has zero planes present
        pooled = (feats * mask).sum(dim=1) / counts  # (B, feat_dim)

        return self.head(pooled)
