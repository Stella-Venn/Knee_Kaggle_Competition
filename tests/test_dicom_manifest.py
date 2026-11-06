"""Tests for src/data/dicom_manifest.py. Synthetic cases use hand-computed geometry;
the real-data cases run against the 3 visible test studies if they've been downloaded
(data/raw/test_series) and are skipped otherwise.
"""

import glob
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from data.dicom_manifest import (
    build_study_manifest,
    plane_from_orientation,
    read_slice_header,
    slice_normal,
    summarize_series,
)

AXIAL = [1, 0, 0, 0, 1, 0]      # normal (0, 0, 1)
SAGITTAL = [0, 1, 0, 0, 0, -1]  # normal (-1, 0, 0)
CORONAL = [1, 0, 0, 0, 0, -1]   # normal (0, 1, 0)


def header(z, inst, iop=AXIAL, **over):
    h = {
        "PatientID": "p1", "Manufacturer": "M", "ManufacturerModelName": "X",
        "MagneticFieldStrength": 1.5, "Laterality": None, "Rows": 320, "Columns": 320,
        "PixelSpacing": [0.3, 0.3], "ImageOrientationPatient": iop,
        "ImagePositionPatient": [5.0, 7.0, z] if iop == AXIAL else
        ([z, 7.0, 5.0] if iop == SAGITTAL else [5.0, z, 7.0]),
        "InstanceNumber": inst, "NumberOfFrames": None,
    }
    h.update(over)
    return h


def test_plane_from_orientation():
    assert plane_from_orientation(AXIAL) == "Axial"
    assert plane_from_orientation(SAGITTAL) == "Sagittal"
    assert plane_from_orientation(CORONAL) == "Coronal"
    assert list(slice_normal(SAGITTAL)) == [-1.0, 0.0, 0.0]


def test_uniform_series_and_instance_order_mismatch():
    # positions 0,3,6 but instance numbers 1,3,2 -> sorting by InstanceNumber would be wrong
    row = summarize_series([header(0, 1), header(3, 3), header(6, 2)], "s", "r")
    assert row["n_slices"] == 3
    assert row["plane_from_geometry"] == "Axial"
    assert (row["gap_min"], row["gap_median"], row["gap_max"]) == (3.0, 3.0, 3.0)
    assert row["uniform_spacing"] is True
    assert row["instance_order_matches_position"] is False
    assert row["n_unique_positions"] == 3


def test_instance_order_reversed_counts_as_matching():
    row = summarize_series([header(0, 3), header(3, 2), header(6, 1)])
    assert row["instance_order_matches_position"] is True


def test_irregular_gaps_flagged():
    row = summarize_series([header(z, i) for i, z in enumerate([0, 0.6, 1.8, 4.2])])
    assert row["gap_min"] == pytest.approx(0.6)
    assert row["gap_median"] == pytest.approx(1.2)
    assert row["gap_max"] == pytest.approx(2.4)
    assert row["uniform_spacing"] is False


def test_sagittal_positions_use_projection_not_z():
    # sagittal normal is -x; x positions 10, 7, 4 -> projections -10,-7,-4, gaps of 3
    row = summarize_series([header(10, 1, SAGITTAL), header(7, 2, SAGITTAL), header(4, 3, SAGITTAL)])
    assert row["plane_from_geometry"] == "Sagittal"
    assert row["gap_median"] == pytest.approx(3.0)
    assert row["uniform_spacing"] is True


def test_duplicate_positions_counted():
    row = summarize_series([header(0, 1), header(0, 2), header(3, 3)])
    assert row["n_unique_positions"] == 2


def test_inconsistent_shape_and_patient_flagged():
    row = summarize_series([header(0, 1), header(3, 2, Rows=256, PatientID="p2")])
    assert row["consistent_shape"] is False
    assert row["consistent_patient"] is False
    assert row["consistent_orientation"] is True


def test_missing_geometry_gives_none_not_crash():
    row = summarize_series([header(0, 1, ImagePositionPatient=None)])
    assert row["has_geometry"] is False
    assert row["plane_from_geometry"] is None and row["gap_median"] is None


def test_single_slice_series():
    row = summarize_series([header(0, 1)])
    assert row["n_slices"] == 1 and row["gap_median"] is None and row["n_unique_positions"] == 1


def test_study_manifest_flags_mixed_scanners_and_patients():
    a = summarize_series([header(0, 1)], "study1", "a")
    b = summarize_series([header(0, 1, Manufacturer="Other", PatientID="p2", Laterality="L")], "study1", "b")
    c = summarize_series([header(0, 1)], "study2", "c")
    out = {r["StudyInstanceUID"]: r for r in build_study_manifest([a, b, c])}
    assert out["study1"]["n_series"] == 2 and out["study1"]["n_patient_ids"] == 2
    assert out["study1"]["scanner_consistent"] is False
    assert out["study1"]["Laterality"] == "L"
    assert out["study2"]["scanner_consistent"] is True and out["study2"]["Laterality"] is None


# --- real files -------------------------------------------------------------

REAL = sorted(glob.glob(str(ROOT / "data/raw/test_series/*/*/*.dcm")))
needs_real = pytest.mark.skipif(not REAL, reason="test DICOMs not downloaded")


@needs_real
def test_real_header_types_and_series_summary():
    by_series = {}
    for path in REAL:
        by_series.setdefault(Path(path).parent, []).append(read_slice_header(path))
    axial_pd = next(hs for hs in by_series.values() if hs[0]["SeriesDescription"] == "pd_tse_tra_d")
    row = summarize_series(axial_pd)
    assert row["n_slices"] == 34 and row["plane_from_geometry"] == "Axial"
    assert row["Rows"] == 960 and row["Columns"] == 960
    assert row["PixelSpacingRow"] == pytest.approx(0.1562)
    assert row["gap_median"] == pytest.approx(3.3, abs=1e-3) and row["uniform_spacing"] is True
    # In all 5 visible series InstanceNumber is monotone in position (some reversed, none
    # scrambled). Scrambled order needs the full training set to show up; see the manifest.
    assert row["instance_order_matches_position"] is True
    assert row["PatientID"] == "857c0f7e-c11" and row["Laterality"] == "L"
    assert row["MagneticFieldStrength"] == 1.5 and row["multiframe"] is False


@needs_real
def test_real_thinned_3d_series_is_irregular():
    hs = [read_slice_header(p) for p in REAL if "t2_de3d" in (read_slice_header(p)["SeriesDescription"] or "")]
    row = summarize_series(hs)
    assert row["n_slices"] == 73 and row["uniform_spacing"] is False
    assert row["gap_min"] == pytest.approx(0.6, abs=0.01)
