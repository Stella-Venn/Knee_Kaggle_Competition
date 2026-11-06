"""DICOM header manifest: what is actually in the imaging files, at scale.

Pixel data is never read here. `read_slice_header` pulls the tags we need from
one .dcm file; `summarize_series` turns one series' slice headers into a single
manifest row (geometry, spacing regularity, scanner, patient id);
`build_study_manifest` rolls series rows up to one row per study and flags any
disagreement between a study's series.

Deliberately dependency-light (pydicom + numpy + stdlib) and importable on its
own: scripts/build_manifest_kernel.py embeds this file's source into a Kaggle
kernel, which is where it runs over the full ~430 GB training set (too big to
hold locally).

Findings on the 3 visible test studies (2026-09-21) that shaped what is recorded:
- Rows/Columns vary a lot between series (640x640 ... 640x1280, non-square) and
  PixelSpacing between 0.16 and 0.25 mm, so pixel size cannot be assumed.
- InstanceNumber was monotone in position in all 5 visible series (direction varies),
  so it happens to be safe there. Whether that holds across the full training set is
  exactly what `instance_order_matches_position` measures; until it's confirmed,
  slices are sorted by projected ImagePositionPatient, not InstanceNumber.
- One 3D axial series had irregular gaps (0.6-4.2 mm): slices were dropped from a
  0.6 mm volume. Spacing regularity is therefore recorded, not assumed.
- Laterality was populated ('L') there, though empty in the one earlier training
  sample -- so it is recorded per series rather than assumed empty.
"""

import os
from collections import Counter

import numpy as np

HEADER_FIELDS = [
    "PatientID", "Manufacturer", "ManufacturerModelName", "MagneticFieldStrength",
    "Laterality", "ImageLaterality", "BodyPartExamined", "SeriesDescription",
    "Rows", "Columns", "PixelSpacing", "SliceThickness", "SpacingBetweenSlices",
    "RepetitionTime", "EchoTime", "ImageOrientationPatient", "ImagePositionPatient",
    "InstanceNumber", "WindowCenter", "WindowWidth", "BitsStored",
    "PhotometricInterpretation", "RescaleSlope", "RescaleIntercept",
    "NumberOfFrames", "PatientPosition", "PatientSex",
]
_LIST_FIELDS = {"PixelSpacing", "ImageOrientationPatient", "ImagePositionPatient"}
_NUMERIC_FIELDS = {
    "MagneticFieldStrength", "Rows", "Columns", "SliceThickness", "SpacingBetweenSlices",
    "RepetitionTime", "EchoTime", "InstanceNumber", "WindowCenter", "WindowWidth",
    "BitsStored", "RescaleSlope", "RescaleIntercept", "NumberOfFrames",
}
PLANE_BY_NORMAL_AXIS = {0: "Sagittal", 1: "Coronal", 2: "Axial"}

# Slice gaps within this fraction of the median gap count as "uniform".
UNIFORM_GAP_TOLERANCE = 0.02


def _plain(value):
    """Convert pydicom value types (DSfloat, MultiValue, ...) to plain Python."""
    if value is None:
        return None
    if isinstance(value, (str, bytes)):
        return str(value).strip()
    try:
        return [float(v) for v in value]
    except TypeError:
        return float(value)


def _to_float(raw):
    """float from a scalar or multi-valued DICOM number (first value); None if
    absent, empty, or non-numeric (e.g. a MagneticFieldStrength stored as text).
    """
    if raw is None or isinstance(raw, str) and not raw.strip():
        return None
    if not isinstance(raw, (str, bytes)) and hasattr(raw, "__len__"):
        if len(raw) == 0:
            return None
        raw = raw[0]
    try:
        return float(raw)
    except (TypeError, ValueError):
        return None


def read_slice_header(path) -> dict:
    """Header fields of one .dcm file, as plain Python. A missing or empty tag is
    None. Never reads pixel data.
    """
    import pydicom

    ds = pydicom.dcmread(path, stop_before_pixels=True)
    out = {"path": str(path), "SOPInstanceUID": os.path.splitext(os.path.basename(path))[0]}
    for name in HEADER_FIELDS:
        raw = getattr(ds, name, None)
        if name in _LIST_FIELDS:
            out[name] = None if raw is None else [float(v) for v in raw]
        elif name in _NUMERIC_FIELDS:
            out[name] = _to_float(raw)
        else:
            value = _plain(raw)
            out[name] = None if value in ("", [], None) else value
    return out


def slice_normal(iop):
    """Unit normal of the image plane from ImageOrientationPatient (6 numbers)."""
    iop = np.asarray(iop, dtype=float)
    n = np.cross(iop[:3], iop[3:])
    return n / np.linalg.norm(n)


def plane_from_orientation(iop) -> str:
    """Sagittal/Coronal/Axial from which patient axis the slice normal is closest to."""
    return PLANE_BY_NORMAL_AXIS[int(np.argmax(np.abs(slice_normal(iop))))]


def _consistent(values):
    """Whether all non-None values are identical (rounded floats)."""
    seen = {tuple(np.round(v, 4)) if isinstance(v, (list, tuple)) else v for v in values}
    return len(seen) <= 1


def _first(headers, key):
    return next((h[key] for h in headers if h.get(key) is not None), None)


def summarize_series(headers, study_id=None, series_id=None) -> dict:
    """One manifest row from all slice headers of a series. Slices are ordered by
    position along the slice normal, never by filename or InstanceNumber.
    """
    n = len(headers)
    row = {"StudyInstanceUID": study_id, "SeriesInstanceUID": series_id, "n_slices": n}
    for key in ("PatientID", "Manufacturer", "ManufacturerModelName", "MagneticFieldStrength",
                "Laterality", "ImageLaterality", "BodyPartExamined", "SeriesDescription",
                "PatientPosition", "PatientSex", "PhotometricInterpretation",
                "SliceThickness", "SpacingBetweenSlices", "RepetitionTime", "EchoTime",
                "BitsStored", "WindowCenter", "WindowWidth", "RescaleSlope", "RescaleIntercept"):
        row[key] = _first(headers, key)
    row["Rows"], row["Columns"] = _first(headers, "Rows"), _first(headers, "Columns")
    ps = _first(headers, "PixelSpacing")
    row["PixelSpacingRow"], row["PixelSpacingCol"] = (ps[0], ps[1]) if ps else (None, None)
    row["multiframe"] = any((h.get("NumberOfFrames") or 1) > 1 for h in headers)

    row["consistent_shape"] = _consistent([(h["Rows"], h["Columns"]) for h in headers])
    row["consistent_pixel_spacing"] = _consistent([h.get("PixelSpacing") for h in headers])
    row["consistent_orientation"] = _consistent([h.get("ImageOrientationPatient") for h in headers])
    row["consistent_patient"] = _consistent([h.get("PatientID") for h in headers])

    has_geometry = all(h.get("ImageOrientationPatient") and h.get("ImagePositionPatient") for h in headers)
    row["has_geometry"] = has_geometry
    row["plane_from_geometry"] = None
    for key in ("gap_min", "gap_median", "gap_max", "uniform_spacing", "n_unique_positions",
                "instance_order_matches_position"):
        row[key] = None
    if not has_geometry or n == 0:
        return row

    iop = headers[0]["ImageOrientationPatient"]
    normal = slice_normal(iop)
    row["plane_from_geometry"] = plane_from_orientation(iop)
    positions = np.array([np.dot(h["ImagePositionPatient"], normal) for h in headers])
    order = np.argsort(positions, kind="stable")
    sorted_pos = positions[order]
    row["n_unique_positions"] = int(len(np.unique(np.round(sorted_pos, 3))))
    if n > 1:
        gaps = np.diff(sorted_pos)
        row["gap_min"], row["gap_median"], row["gap_max"] = (
            float(gaps.min()), float(np.median(gaps)), float(gaps.max()),
        )
        med = float(np.median(gaps))
        row["uniform_spacing"] = bool(med > 0 and np.all(np.abs(gaps - med) <= UNIFORM_GAP_TOLERANCE * med))
    inst = [h.get("InstanceNumber") for h in headers]
    if all(v is not None for v in inst):
        by_position = [inst[i] for i in order]
        row["instance_order_matches_position"] = bool(
            by_position == sorted(inst) or by_position == sorted(inst, reverse=True)
        )
    return row


def build_study_manifest(series_rows):
    """One row per study from series rows, with disagreement flags. A study's
    scanner/patient fields should agree across its series; `n_patient_ids` > 1 or
    `scanner_consistent` False would mean the study mixes sources.
    """
    by_study = {}
    for r in series_rows:
        by_study.setdefault(r["StudyInstanceUID"], []).append(r)
    out = []
    for study, rows in by_study.items():
        patients = {r["PatientID"] for r in rows if r["PatientID"]}
        scanners = {(r["Manufacturer"], r["ManufacturerModelName"], r["MagneticFieldStrength"]) for r in rows}
        lat = Counter(r["Laterality"] for r in rows if r["Laterality"])
        out.append({
            "StudyInstanceUID": study,
            "PatientID": sorted(patients)[0] if patients else None,
            "n_patient_ids": len(patients),
            "Manufacturer": rows[0]["Manufacturer"],
            "ManufacturerModelName": rows[0]["ManufacturerModelName"],
            "MagneticFieldStrength": rows[0]["MagneticFieldStrength"],
            "scanner_consistent": len(scanners) == 1,
            "Laterality": lat.most_common(1)[0][0] if lat else None,
            "n_series": len(rows),
            "n_slices_total": sum(r["n_slices"] for r in rows),
        })
    return out


def series_slice_paths(root, study_id, series_id):
    """Sorted .dcm paths of one series directory."""
    d = os.path.join(root, study_id, series_id)
    return sorted(os.path.join(d, f) for f in os.listdir(d) if f.endswith(".dcm"))
