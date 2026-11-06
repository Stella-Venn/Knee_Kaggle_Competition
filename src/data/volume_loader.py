"""DICOM -> volume loader (gameplan module 4).

Scope, on purpose: this module reconstructs one 3D volume per *series*, in its
own native pixel spacing, and provides generic resample/normalize primitives.
It does NOT decide how a study's 3-14 series get combined into one model
input (which series to keep, how to pad/stack a variable count) or what
target spacing/shape a model should consume -- those are dataset/sampler
(module 6) and model architecture (module 7) decisions, made once there's a
model to make them for. `resample_volume` therefore takes `target_spacing`
as a required argument rather than a guessed default.

Geometry handling is grounded in the full-dataset manifest
(data/data/processed/manifest/, see [redacted: private project notes] "DICOM header fields" and
`dicom_manifest.py`), not a single sample file:
- Slices are sorted by projected `ImagePositionPatient` (dataset-wide this
  agreed with `InstanceNumber` order on all 24371 series, but sorting by
  position is what's actually correct and costs nothing).
- `consistent_orientation` is False for only 13/24371 series. Rather than
  crash a full-dataset caching run on those 13, `load_series_volume` uses the
  first slice's orientation as the reference plane and flags
  `orientation_consistent=False` in its result so a caller can choose to skip
  or accept it.
- No multiframe series exist in this dataset (`NumberOfFrames` never >1), so
  one file is always one 2D slice; not handled here, would need a new code
  [redacted: detail about the data challenge this code was written for]
- `PhotometricInterpretation` is `MONOCHROME2` dataset-wide (no
  MONOCHROME1 inversion to handle).
- Pixel values are rescaled (`RescaleSlope`/`RescaleIntercept`) but NOT
  windowed (`WindowCenter`/`WindowWidth`) here -- windowing is a display
  convenience for radiologists, not a normalization scheme; MRI has no fixed
  unit scale like CT's Hounsfield units, so `normalize_intensity`'s
  per-volume percentile clip+rescale is used instead.
"""

from dataclasses import dataclass, field

import numpy as np

from data.dicom_manifest import plane_from_orientation, read_slice_header, slice_normal


@dataclass
class SeriesVolume:
    volume: np.ndarray  # (D, H, W) float32, rescaled pixel values, native spacing
    slice_spacing: float  # mm, along the slice normal
    row_spacing: float  # mm
    col_spacing: float  # mm
    plane: str  # "Sagittal" / "Coronal" / "Axial", from geometry
    sop_instance_uids: list  # in the same order as `volume`'s first axis
    orientation_consistent: bool
    uniform_spacing: bool
    n_slices: int = field(init=False)

    def __post_init__(self):
        self.n_slices = self.volume.shape[0]


def _rescale(pixels, header) -> np.ndarray:
    slope = header.get("RescaleSlope")
    intercept = header.get("RescaleIntercept")
    out = pixels.astype(np.float32)
    if slope not in (None, 1.0):
        out = out * float(slope)
    if intercept not in (None, 0.0):
        out = out + float(intercept)
    return out


def read_slice_pixels(path):
    """(rescaled float32 2D array, header dict) for one .dcm file. Header is
    the same shape as dicom_manifest.read_slice_header's output.
    """
    import pydicom

    ds = pydicom.dcmread(path)
    header = read_slice_header(path)
    pixels = _rescale(ds.pixel_array, header)
    return pixels, header


def load_series_volume(series_dir) -> SeriesVolume:
    """Read every .dcm directly inside `series_dir` (str or Path) and stack
    them into one volume, sorted by physical position along the slice normal
    (never by filename or InstanceNumber -- see module docstring). Raises
    ValueError if slices don't share a common shape (stacking would silently
    be wrong otherwise) or if geometry tags are missing.
    """
    import os

    paths = sorted(
        os.path.join(str(series_dir), f) for f in os.listdir(series_dir) if f.endswith(".dcm")
    )
    if not paths:
        raise ValueError(f"no .dcm files found in {series_dir}")

    headers = [read_slice_header(p) for p in paths]
    if not all(h.get("ImageOrientationPatient") and h.get("ImagePositionPatient") for h in headers):
        raise ValueError(f"missing ImageOrientationPatient/ImagePositionPatient in {series_dir}")

    shapes = {(h["Rows"], h["Columns"]) for h in headers}
    if len(shapes) > 1:
        raise ValueError(f"inconsistent slice shapes in {series_dir}: {shapes}")

    orientations = [tuple(np.round(h["ImageOrientationPatient"], 4)) for h in headers]
    orientation_consistent = len(set(orientations)) == 1
    reference_iop = headers[0]["ImageOrientationPatient"]  # used regardless; see module docstring
    normal = slice_normal(reference_iop)
    plane = plane_from_orientation(reference_iop)

    positions = np.array([np.dot(h["ImagePositionPatient"], normal) for h in headers])
    order = np.argsort(positions, kind="stable")

    sorted_positions = positions[order]
    uniform_spacing = True
    slice_spacing = None
    if len(sorted_positions) > 1:
        gaps = np.diff(sorted_positions)
        slice_spacing = float(np.median(np.abs(gaps)))
        med = slice_spacing if slice_spacing > 0 else 1.0
        uniform_spacing = bool(np.all(np.abs(np.abs(gaps) - slice_spacing) <= 0.02 * med))
    else:
        slice_spacing = float(headers[0].get("SpacingBetweenSlices") or headers[0].get("SliceThickness") or 1.0)

    ps = headers[0].get("PixelSpacing") or [1.0, 1.0]
    row_spacing, col_spacing = float(ps[0]), float(ps[1])

    slices = []
    sop_uids = []
    for i in order:
        pixels, _ = read_slice_pixels(paths[i])
        slices.append(pixels)
        sop_uids.append(headers[i]["SOPInstanceUID"])
    volume = np.stack(slices, axis=0).astype(np.float32)

    return SeriesVolume(
        volume=volume,
        slice_spacing=slice_spacing,
        row_spacing=row_spacing,
        col_spacing=col_spacing,
        plane=plane,
        sop_instance_uids=sop_uids,
        orientation_consistent=orientation_consistent,
        uniform_spacing=uniform_spacing,
    )


def resample_volume(volume, spacing, target_spacing, order=1):
    """Resample a (D, H, W) volume from `spacing` (dz, dy, dx) mm to
    `target_spacing` mm, by physical size (not by target pixel count) --
    the dataset's Rows/Columns/slice-count vary too much (256-1024px,
    0.073-1.17mm spacing; see [redacted: private project notes]) for a fixed-pixel resize to leave
    a consistent real-world field of view. `order`: scipy.ndimage.zoom
    interpolation order (1 = trilinear, default; 0 = nearest, for masks).
    """
    from scipy.ndimage import zoom

    spacing = np.asarray(spacing, dtype=float)
    target_spacing = np.asarray(target_spacing, dtype=float)
    if spacing.shape != (3,) or target_spacing.shape != (3,):
        raise ValueError("spacing and target_spacing must both be (dz, dy, dx)")
    if np.any(spacing <= 0) or np.any(target_spacing <= 0):
        raise ValueError("spacing values must be positive")
    zoom_factors = spacing / target_spacing
    resampled = zoom(volume, zoom_factors, order=order)
    achieved_spacing = spacing * np.array(volume.shape) / np.array(resampled.shape)
    return resampled.astype(np.float32), tuple(achieved_spacing)


def normalize_intensity(volume, low_pct=0.5, high_pct=99.5):
    """Per-volume percentile clip + rescale to [0, 1] float32. MRI intensities
    have no fixed unit scale (unlike CT Hounsfield units), so this -- not a
    fixed window -- is the normalization; percentiles (not min/max) so a few
    bright artifact voxels don't compress the useful range. A volume with no
    intensity spread (low == high, e.g. an all-zero or synthetic constant
    volume) returns all zeros rather than dividing by zero.
    """
    volume = np.asarray(volume, dtype=np.float32)
    if not 0 <= low_pct < high_pct <= 100:
        raise ValueError(f"need 0 <= low_pct < high_pct <= 100, got {low_pct}, {high_pct}")
    low, high = np.percentile(volume, [low_pct, high_pct])
    if high <= low:
        return np.zeros_like(volume, dtype=np.float32)
    clipped = np.clip(volume, low, high)
    return ((clipped - low) / (high - low)).astype(np.float32)


def center_crop_or_pad(volume, target_shape, pad_value=0.0):
    """Crop or pad a (D, H, W) volume to exactly `target_shape`, independently
    per axis, centered. This is a batching necessity (a DataLoader needs every
    sample the same size) kept separate from `resample_volume` on purpose:
    resampling fixes physical scale (mm/voxel, a property of the data),
    cropping/padding fixes tensor size (a property of what a given training
    run wants to batch) -- baking a target shape into the resample step would
    mean re-running that expensive op for every shape experiment instead of
    just re-cropping an already-resampled, already-cached volume.
    """
    volume = np.asarray(volume, dtype=np.float32)
    if len(target_shape) != volume.ndim:
        raise ValueError(f"target_shape {target_shape} must have {volume.ndim} dims, got {len(target_shape)}")
    out = volume
    for axis, target in enumerate(target_shape):
        current = out.shape[axis]
        if current > target:
            start = (current - target) // 2
            out = np.take(out, range(start, start + target), axis=axis)
        elif current < target:
            before = (target - current) // 2
            after = target - current - before
            pad_width = [(0, 0)] * out.ndim
            pad_width[axis] = (before, after)
            out = np.pad(out, pad_width, mode="constant", constant_values=pad_value)
    return out.astype(np.float32)
