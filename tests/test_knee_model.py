"""Tests for src/models/knee_model.py. The masked-pooling tests are the load-
bearing ones -- see knee_model.py's module docstring for why masking has to
exclude a missing plane from the backbone call itself, not just from the
pooled sum after the fact.
"""

import glob
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from models.backbone3d import Simple3DCNN
from models.knee_model import KneeMultiViewModel


def _tiny_model(n_classes=12, hidden_dim=None):
    return KneeMultiViewModel(
        n_classes=n_classes, hidden_dim=hidden_dim,
        in_channels=1, stem_channels=2, channels=(4,),
    )


def test_output_shape():
    model = _tiny_model()
    image = torch.randn(2, 3, 8, 8, 8)
    plane_present = torch.ones(2, 3, dtype=torch.bool)
    out = model(image, plane_present)
    assert out.shape == (2, 12)


def test_output_shape_with_hidden_layer():
    model = _tiny_model(hidden_dim=16)
    out = model(torch.randn(1, 3, 8, 8, 8), torch.ones(1, 3, dtype=torch.bool))
    assert out.shape == (1, 12)


def test_masked_plane_content_does_not_affect_output():
    model = _tiny_model()
    model.eval()
    image = torch.randn(1, 3, 8, 8, 8)
    plane_present = torch.tensor([[True, True, False]])

    image_garbage = image.clone()
    image_garbage[0, 2] = torch.randn(8, 8, 8) * 1000  # wildly different content, same "missing" flag

    with torch.no_grad():
        out1 = model(image, plane_present)
        out2 = model(image_garbage, plane_present)
    torch.testing.assert_close(out1, out2)


def test_all_planes_present_matches_manual_average_pooling():
    torch.manual_seed(0)
    backbone = Simple3DCNN(in_channels=1, stem_channels=2, channels=(4,))
    model = KneeMultiViewModel(n_classes=12, backbone=backbone)
    model.eval()

    image = torch.randn(2, 3, 8, 8, 8)
    plane_present = torch.ones(2, 3, dtype=torch.bool)

    with torch.no_grad():
        out = model(image, plane_present)
        # manual: run backbone on each plane separately, average, then the head
        feats = torch.stack([backbone(image[:, p:p+1].reshape(2, 1, 8, 8, 8)) for p in range(3)], dim=1)
        manual_pooled = feats.mean(dim=1)
        manual_out = model.head(manual_pooled)

    torch.testing.assert_close(out, manual_out)


def test_partial_planes_matches_manual_average_of_present_only():
    torch.manual_seed(0)
    backbone = Simple3DCNN(in_channels=1, stem_channels=2, channels=(4,))
    model = KneeMultiViewModel(n_classes=12, backbone=backbone)
    model.eval()

    image = torch.randn(1, 3, 8, 8, 8)
    plane_present = torch.tensor([[True, False, True]])

    with torch.no_grad():
        out = model(image, plane_present)
        f0 = backbone(image[:, 0:1])
        f2 = backbone(image[:, 2:3])
        manual_pooled = (f0 + f2) / 2
        manual_out = model.head(manual_pooled)

    torch.testing.assert_close(out, manual_out)


def test_study_with_zero_planes_present_does_not_crash_or_nan():
    model = _tiny_model()
    image = torch.randn(1, 3, 8, 8, 8)
    plane_present = torch.zeros(1, 3, dtype=torch.bool)
    out = model(image, plane_present)
    assert out.shape == (1, 12)
    assert torch.isfinite(out).all()


def test_gradients_flow_with_mixed_batch_masks():
    model = _tiny_model()
    image = torch.randn(4, 3, 8, 8, 8, requires_grad=False)
    plane_present = torch.tensor([
        [True, True, True], [True, False, True],
        [False, False, True], [True, True, False],
    ])
    out = model(image, plane_present)
    loss = out.sum()
    loss.backward()
    grads = [p.grad for p in model.parameters() if p.requires_grad]
    assert all(g is not None for g in grads)
    assert all(torch.isfinite(g).all() for g in grads)
    assert any(g.abs().sum() > 0 for g in grads)  # not all-zero gradients


def test_rejects_plane_present_shape_mismatch():
    model = _tiny_model()
    with pytest.raises(ValueError):
        model(torch.randn(2, 3, 8, 8, 8), torch.ones(2, 2, dtype=torch.bool))


# --- real data: KneeStudyDataset -> KneeMultiViewModel end to end -------------

REAL_SERIES_DIRS = sorted({str(Path(p).parent) for p in glob.glob(str(Path(__file__).resolve().parents[1] / "data/raw/test_series/*/*/*.dcm"))})
needs_real = pytest.mark.skipif(not REAL_SERIES_DIRS, reason="test DICOMs not downloaded")


@needs_real
def test_real_dataset_item_through_model(tmp_path):
    from data.dataset import KneeStudyDataset
    from data.label_mining.final_labels import MASKED
    from data.volume_loader import load_series_volume, normalize_intensity, resample_volume
    from utils.constants import LABEL_COLUMNS
    import numpy as np

    study_dir = Path(REAL_SERIES_DIRS[0]).parent
    study_id = study_dir.name
    series_rows = []
    for series_dir in sorted(p for p in study_dir.iterdir() if p.is_dir()):
        result = load_series_volume(series_dir)
        resampled, _ = resample_volume(
            result.volume, (result.slice_spacing, result.row_spacing, result.col_spacing),
            target_spacing=(3.0, 0.5, 0.5),
        )
        normalized = normalize_intensity(resampled)
        out_dir = tmp_path / study_id
        out_dir.mkdir(exist_ok=True)
        np.save(out_dir / f"{series_dir.name}.npy", normalized)
        series_rows.append({
            "SeriesInstanceUID": series_dir.name, "n_slices": result.n_slices,
            "csv_Anatomical_Plane": result.plane,
        })

    labels = np.full((1, len(LABEL_COLUMNS)), MASKED, dtype=np.int8)
    ds = KneeStudyDataset([study_id], labels, {study_id: series_rows}, cache_dir=tmp_path,
                           target_shape=(16, 64, 64))
    item = ds[0]

    model = _tiny_model()
    model.eval()
    with torch.no_grad():
        logits = model(item["image"].unsqueeze(0), item["plane_present"].unsqueeze(0))
    assert logits.shape == (1, 12)
    assert torch.isfinite(logits).all()



def test_forward_when_backbone_returns_half_precision():
    """Regression (2026-09-25): under CUDA autocast the backbone returns
    float16, and writing that into a feature tensor pre-allocated in the
    INPUT's dtype (float32) crashed the first full GPU run ("Index put
    requires the source and destination dtypes match"). CPU autocast can't
    reproduce it (it leaves Conv3d in float32), so a stub backbone returns
    half precision, exactly as the real one does on the GPU."""
    import torch
    from models.knee_model import KneeMultiViewModel

    class HalfBackbone(torch.nn.Module):
        feature_dim = 8

        def __init__(self):
            super().__init__()
            self.proj = torch.nn.Linear(1, 8)

        def forward(self, x):  # (N, 1, D, H, W) -> (N, 8), half precision like autocast
            return self.proj(x.mean(dim=(2, 3, 4))).half()

    model = KneeMultiViewModel(n_classes=12, backbone=HalfBackbone())
    image = torch.rand(2, 3, 4, 8, 8)
    present = torch.tensor([[True, False, True], [True, True, True]])
    feats = model.backbone(image.reshape(6, 1, 4, 8, 8))
    assert feats.dtype == torch.float16  # the condition that crashed
    model.head = model.head.half()  # stands in for autocast running the head in half
    logits = model(image, present)
    assert logits.shape == (2, 12)
    assert torch.isfinite(logits.float()).all()
    logits.float().sum().backward()  # gradients flow through the index_put
    assert model.backbone.proj.weight.grad is not None
