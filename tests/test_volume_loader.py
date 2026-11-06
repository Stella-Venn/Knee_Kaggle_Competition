"""Tests for src/data/volume_loader.py. Synthetic DICOM-like fixtures for the
pure-numpy logic (resample/normalize, and geometry via a small in-memory
series built with pydicom); real-file tests against the one downloaded test
series (data/raw/test_series/), skipped if absent.
"""

import glob
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from data.volume_loader import (
    center_crop_or_pad,
    load_series_volume,
    normalize_intensity,
    resample_volume,
)


# --- normalize_intensity -----------------------------------------------------

def test_normalize_clips_and_rescales_to_unit_range():
    v = np.array([0, 10, 50, 90, 100], dtype=np.float32)
    out = normalize_intensity(v, low_pct=0, high_pct=100)
    assert out.min() == pytest.approx(0.0) and out.max() == pytest.approx(1.0)
    assert out[2] == pytest.approx(0.5, abs=0.01)


def test_normalize_clips_outlier_beyond_percentile():
    v = np.array([0.0] * 98 + [1000.0] + [1.0], dtype=np.float32)
    out = normalize_intensity(v, low_pct=0.5, high_pct=99.5)
    assert out.max() <= 1.0  # the 1000-outlier gets clipped, not stretching the range
    assert out[-2] == pytest.approx(1.0)  # the clipped outlier lands at the top


def test_normalize_constant_volume_returns_zeros_not_nan():
    v = np.full((4, 4), 7.0, dtype=np.float32)
    out = normalize_intensity(v)
    assert np.all(out == 0.0)


def test_normalize_rejects_bad_percentiles():
    with pytest.raises(ValueError):
        normalize_intensity(np.zeros(3), low_pct=50, high_pct=10)


# --- resample_volume ----------------------------------------------------------

def test_resample_doubles_size_when_spacing_halves():
    v = np.zeros((10, 20, 20), dtype=np.float32)
    out, achieved = resample_volume(v, spacing=(4.0, 1.0, 1.0), target_spacing=(2.0, 1.0, 1.0))
    assert out.shape[0] == pytest.approx(20, abs=1)
    assert out.shape[1:] == (20, 20)
    assert achieved[0] == pytest.approx(4.0 * 10 / out.shape[0], rel=0.1)


def test_resample_identity_spacing_preserves_shape():
    v = np.random.default_rng(0).random((5, 8, 8)).astype(np.float32)
    out, achieved = resample_volume(v, spacing=(3.0, 0.5, 0.5), target_spacing=(3.0, 0.5, 0.5))
    assert out.shape == v.shape
    assert achieved == pytest.approx((3.0, 0.5, 0.5))


def test_resample_rejects_bad_spacing_shape():
    with pytest.raises(ValueError):
        resample_volume(np.zeros((2, 2, 2)), spacing=(1.0, 1.0), target_spacing=(1.0, 1.0, 1.0))


def test_resample_rejects_nonpositive_spacing():
    with pytest.raises(ValueError):
        resample_volume(np.zeros((2, 2, 2)), spacing=(0.0, 1.0, 1.0), target_spacing=(1.0, 1.0, 1.0))


# --- center_crop_or_pad --------------------------------------------------------

def test_crop_or_pad_crops_centered():
    v = np.arange(10, dtype=np.float32).reshape(1, 1, 10)
    out = center_crop_or_pad(v, (1, 1, 4))
    assert out.shape == (1, 1, 4)
    assert list(out[0, 0]) == [3.0, 4.0, 5.0, 6.0]  # centered 4 of 10


def test_crop_or_pad_pads_centered_with_value():
    v = np.ones((1, 1, 4), dtype=np.float32)
    out = center_crop_or_pad(v, (1, 1, 8), pad_value=-1.0)
    assert out.shape == (1, 1, 8)
    assert list(out[0, 0]) == [-1, -1, 1, 1, 1, 1, -1, -1]


def test_crop_or_pad_mixed_axes_and_noop():
    v = np.ones((5, 3, 3), dtype=np.float32)
    out = center_crop_or_pad(v, (3, 3, 5))  # crop axis 0, same axis 1, pad axis 2
    assert out.shape == (3, 3, 5)


def test_crop_or_pad_rejects_wrong_ndim():
    with pytest.raises(ValueError):
        center_crop_or_pad(np.zeros((2, 2, 2)), (2, 2))


# --- load_series_volume: synthetic in-memory DICOM series ---------------------

def _write_synthetic_series(tmp_path, n=5, z_step=3.0, iop=(1, 0, 0, 0, 1, 0), scramble_filenames=True):
    """A minimal valid multi-file DICOM series on disk, axial, z ascending by
    z_step per slice, InstanceNumber deliberately reversed vs. position so a
    filename/instance-based sort would get slice order wrong.
    """
    import pydicom
    from pydicom.dataset import FileDataset, FileMetaDataset
    from pydicom.uid import ExplicitVRLittleEndian, generate_uid

    for i in range(n):
        meta = FileMetaDataset()
        meta.MediaStorageSOPClassUID = generate_uid()
        meta.MediaStorageSOPInstanceUID = generate_uid()
        meta.TransferSyntaxUID = ExplicitVRLittleEndian
        ds = FileDataset(None, {}, file_meta=meta, preamble=b"\0" * 128)
        ds.SOPInstanceUID = meta.MediaStorageSOPInstanceUID
        ds.Rows, ds.Columns = 4, 4
        ds.PixelSpacing = [0.5, 0.5]
        ds.ImageOrientationPatient = list(iop)
        ds.ImagePositionPatient = [0.0, 0.0, i * z_step]
        ds.InstanceNumber = n - i  # reversed vs. position, on purpose
        ds.SamplesPerPixel = 1
        ds.PhotometricInterpretation = "MONOCHROME2"
        ds.BitsAllocated = 16
        ds.BitsStored = 16
        ds.HighBit = 15
        ds.PixelRepresentation = 0
        ds.RescaleSlope = "2"
        ds.RescaleIntercept = "10"
        # pixel value == slice index (before rescale) so read-back order is checkable
        ds.PixelData = (np.full((4, 4), i, dtype=np.uint16)).tobytes()
        # scramble filenames so a filename-sort would not match position order
        name = f"z{'reversed' if scramble_filenames else i}_{n - i}.dcm"
        pydicom.dcmwrite(tmp_path / name, ds, enforce_file_format=True, little_endian=True, implicit_vr=False)
    return tmp_path


def test_load_series_volume_sorts_by_position_not_filename_or_instance(tmp_path):
    _write_synthetic_series(tmp_path, n=5, z_step=3.0)
    result = load_series_volume(tmp_path)
    assert result.n_slices == 5
    assert result.volume.shape == (5, 4, 4)
    # rescale applied: pixel value i -> 2*i + 10; first slice (z=0, pixel value 0) must be first
    assert list(result.volume[:, 0, 0]) == [10.0, 12.0, 14.0, 16.0, 18.0]
    assert result.slice_spacing == pytest.approx(3.0)
    assert result.uniform_spacing is True
    assert result.plane == "Axial"
    assert result.row_spacing == pytest.approx(0.5) and result.col_spacing == pytest.approx(0.5)
    assert result.orientation_consistent is True


def test_load_series_volume_single_slice(tmp_path):
    _write_synthetic_series(tmp_path, n=1)
    result = load_series_volume(tmp_path)
    assert result.n_slices == 1
    assert result.volume.shape == (1, 4, 4)


def test_load_series_volume_empty_dir_raises(tmp_path):
    with pytest.raises(ValueError):
        load_series_volume(tmp_path)


# --- real files ---------------------------------------------------------------

REAL_SERIES_DIRS = sorted({str(Path(p).parent) for p in glob.glob(str(ROOT / "data/raw/test_series/*/*/*.dcm"))})
needs_real = pytest.mark.skipif(not REAL_SERIES_DIRS, reason="test DICOMs not downloaded")


@needs_real
def test_real_series_end_to_end_load_resample_normalize():
    # the smallest real series, to keep the test fast
    sizes = [(d, len(list(Path(d).glob("*.dcm")))) for d in REAL_SERIES_DIRS]
    series_dir = min(sizes, key=lambda x: x[1])[0]
    result = load_series_volume(series_dir)
    assert result.volume.ndim == 3 and result.volume.shape[0] == result.n_slices
    assert result.plane in ("Sagittal", "Coronal", "Axial")
    assert result.row_spacing > 0 and result.col_spacing > 0 and result.slice_spacing > 0

    resampled, achieved = resample_volume(
        result.volume,
        spacing=(result.slice_spacing, result.row_spacing, result.col_spacing),
        target_spacing=(3.0, 0.5, 0.5),
    )
    assert resampled.ndim == 3
    assert np.all(np.array(achieved) > 0)

    normalized = normalize_intensity(resampled)
    assert normalized.min() >= 0.0 and normalized.max() <= 1.0 + 1e-5
