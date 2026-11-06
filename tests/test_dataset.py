"""Tests for src/data/dataset.py."""

import glob
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from data.dataset import (
    KneeStudyDataset,
    VolumeCacheMissingError,
    load_cached_volume,
    select_canonical_series,
    spacing_tag,
)
from data.label_mining.final_labels import MASKED
from utils.constants import LABEL_COLUMNS


def test_spacing_tag_formats_cleanly():
    assert spacing_tag((3.0, 0.5, 0.5)) == "3x0.5x0.5"
    assert spacing_tag((1, 1, 1)) == "1x1x1"


# --- select_canonical_series -------------------------------------------------

def test_select_canonical_series_picks_largest_per_plane():
    rows = [
        {"SeriesInstanceUID": "sag_small", "n_slices": 20, "csv_Anatomical_Plane": "Sagittal"},
        {"SeriesInstanceUID": "sag_big", "n_slices": 34, "csv_Anatomical_Plane": "Sagittal"},
        {"SeriesInstanceUID": "cor1", "n_slices": 25, "csv_Anatomical_Plane": "Coronal"},
    ]
    out = select_canonical_series(rows)
    assert out == {"Sagittal": "sag_big", "Coronal": "cor1"}
    assert "Axial" not in out  # no axial series in this study -- absent, not guessed


def test_select_canonical_series_ignores_unknown_plane_values():
    rows = [{"SeriesInstanceUID": "s1", "n_slices": 10, "csv_Anatomical_Plane": "Oblique"}]
    assert select_canonical_series(rows) == {}


def test_select_canonical_series_empty_input():
    assert select_canonical_series([]) == {}


# --- load_cached_volume -------------------------------------------------------

def test_load_cached_volume_round_trips(tmp_path):
    (tmp_path / "study1").mkdir()
    arr = np.random.default_rng(0).random((5, 6, 6)).astype(np.float32)
    np.save(tmp_path / "study1" / "series1.npy", arr)
    loaded = load_cached_volume(tmp_path, "study1", "series1")
    np.testing.assert_array_equal(loaded, arr)


def test_load_cached_volume_missing_raises_clear_error(tmp_path):
    with pytest.raises(VolumeCacheMissingError, match="precompute_volume_cache"):
        load_cached_volume(tmp_path, "study1", "missing_series")


# --- KneeStudyDataset: synthetic cache -----------------------------------------

def _make_cache(tmp_path, study_id, series_by_plane):
    """series_by_plane: {plane: (series_id, fill_value)}."""
    d = tmp_path / study_id
    d.mkdir(exist_ok=True)
    for plane, (series_id, fill) in series_by_plane.items():
        np.save(d / f"{series_id}.npy", np.full((4, 8, 8), fill, dtype=np.float32))


def test_dataset_basic_shapes_and_label_mask(tmp_path):
    n_cols = len(LABEL_COLUMNS)
    study_ids = ["s1", "s2"]
    labels = np.full((2, n_cols), MASKED, dtype=np.int8)
    labels[0, 0] = 1
    labels[0, 1] = 0
    labels[1, 2] = 1

    _make_cache(tmp_path, "s1", {
        "Sagittal": ("s1_sag", 0.2), "Coronal": ("s1_cor", 0.4), "Axial": ("s1_ax", 0.6),
    })
    _make_cache(tmp_path, "s2", {"Sagittal": ("s2_sag", 0.9)})  # missing Coronal/Axial

    series_by_study = {
        "s1": [
            {"SeriesInstanceUID": "s1_sag", "n_slices": 30, "csv_Anatomical_Plane": "Sagittal"},
            {"SeriesInstanceUID": "s1_cor", "n_slices": 30, "csv_Anatomical_Plane": "Coronal"},
            {"SeriesInstanceUID": "s1_ax", "n_slices": 30, "csv_Anatomical_Plane": "Axial"},
        ],
        "s2": [
            {"SeriesInstanceUID": "s2_sag", "n_slices": 30, "csv_Anatomical_Plane": "Sagittal"},
        ],
    }

    ds = KneeStudyDataset(study_ids, labels, series_by_study, cache_dir=tmp_path,
                           target_shape=(4, 8, 8))
    assert len(ds) == 2

    item0 = ds[0]
    assert item0["study_id"] == "s1"
    assert item0["image"].shape == (3, 4, 8, 8)
    assert bool(item0["plane_present"].all())
    assert item0["image"][0].mean().item() == pytest.approx(0.2)  # Sagittal channel
    assert item0["image"][1].mean().item() == pytest.approx(0.4)  # Coronal channel
    assert item0["image"][2].mean().item() == pytest.approx(0.6)  # Axial channel
    assert item0["label_mask"][0].item() is True and item0["label"][0].item() == 1.0
    assert item0["label_mask"][2].item() is False  # MASKED column for s1

    item1 = ds[1]
    assert list(item1["plane_present"]) == [True, False, False]  # only Sagittal present
    assert item1["image"][1].sum().item() == 0.0  # missing plane is zero-filled
    assert item1["label_mask"][2].item() is True and item1["label"][2].item() == 1.0


def test_dataset_crops_to_target_shape(tmp_path):
    n_cols = len(LABEL_COLUMNS)
    labels = np.full((1, n_cols), MASKED, dtype=np.int8)
    _make_cache(tmp_path, "s1", {"Sagittal": ("s1_sag", 1.0)})
    series_by_study = {"s1": [
        {"SeriesInstanceUID": "s1_sag", "n_slices": 30, "csv_Anatomical_Plane": "Sagittal"},
    ]}
    ds = KneeStudyDataset(["s1"], labels, series_by_study, cache_dir=tmp_path, target_shape=(2, 4, 4))
    assert ds[0]["image"].shape == (3, 2, 4, 4)  # cache had (4,8,8), cropped down


def test_dataset_rejects_label_shape_mismatch():
    with pytest.raises(ValueError):
        KneeStudyDataset(["s1", "s2"], np.zeros((2, 3)), {}, cache_dir=".")


def test_dataset_missing_cache_file_raises(tmp_path):
    n_cols = len(LABEL_COLUMNS)
    labels = np.full((1, n_cols), MASKED, dtype=np.int8)
    series_by_study = {"s1": [
        {"SeriesInstanceUID": "not_cached", "n_slices": 30, "csv_Anatomical_Plane": "Sagittal"},
    ]}
    ds = KneeStudyDataset(["s1"], labels, series_by_study, cache_dir=tmp_path, target_shape=(4, 8, 8))
    with pytest.raises(VolumeCacheMissingError):
        ds[0]


# --- real data: precompute the cache for the one downloaded test series, ------
# then read it back through KneeStudyDataset end to end.

REAL_SERIES_DIRS = sorted({str(Path(p).parent) for p in glob.glob(str(Path(__file__).resolve().parents[1] / "data/raw/test_series/*/*/*.dcm"))})
needs_real = pytest.mark.skipif(not REAL_SERIES_DIRS, reason="test DICOMs not downloaded")


@needs_real
def test_dataset_end_to_end_with_real_precomputed_cache(tmp_path):
    from data.volume_loader import load_series_volume, normalize_intensity, resample_volume

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
            "SeriesInstanceUID": series_dir.name,
            "n_slices": result.n_slices,
            "csv_Anatomical_Plane": result.plane,
        })

    n_cols = len(LABEL_COLUMNS)
    labels = np.full((1, n_cols), MASKED, dtype=np.int8)
    ds = KneeStudyDataset([study_id], labels, {study_id: series_rows}, cache_dir=tmp_path,
                           target_shape=(16, 64, 64))
    item = ds[0]
    assert item["image"].shape == (3, 16, 64, 64)
    assert item["image"].min().item() >= 0.0 and item["image"].max().item() <= 1.0
    assert bool(item["plane_present"].any())  # this study has at least Sagittal/Coronal/Axial series
