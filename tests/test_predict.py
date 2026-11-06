"""Tests for src/inference/predict.py -- the submission path's plumbing:
every test.csv study gets exactly one valid row, bad data degrades to a
missing plane or a fallback row instead of crashing, and the output passes
the submission validator. A stub model stands in for the real one."""

import csv
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from inference.predict import FALLBACK_PROB, TARGET_SHAPE, preprocess_study, run_inference  # noqa: E402
from inference.validate_submission import validate_submission_file  # noqa: E402
from test_volume_loader import _write_synthetic_series  # noqa: E402
from utils.constants import LABEL_COLUMNS  # noqa: E402


class StubModel(torch.nn.Module):
    """Logit = number of present planes, for every class; records inputs."""

    def __init__(self):
        super().__init__()
        self.calls = []

    def forward(self, image, plane_present):
        self.calls.append((tuple(image.shape), plane_present.tolist()))
        return plane_present.float().sum(dim=1, keepdim=True).repeat(1, len(LABEL_COLUMNS))


def _make_comp(tmp_path):
    comp = tmp_path / "comp"
    good, corrupt, empty = "1.2.3.good", "1.2.3.corrupt", "1.2.3.nodicoms"
    ax = comp / "test_series" / good / "S_AX"; ax.mkdir(parents=True)
    _write_synthetic_series(ax, n=5)
    bad = comp / "test_series" / corrupt / "S_BAD"; bad.mkdir(parents=True)
    (bad / "x.dcm").write_bytes(b"not a dicom")
    with open(comp / "test.csv", "w", newline="") as f:
        csv.writer(f).writerows([["StudyInstanceUID"], [good], [corrupt], [empty]])
    with open(comp / "test_series.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["StudyInstanceUID", "SeriesInstanceUID", "Fluid_Sensitive", "Fat_Suppression", "Anatomical_Plane"])
        w.writerow([good, "S_AX", 0, 0, "Axial"])
        w.writerow([good, "S_MISSING_DIR", 0, 0, "Sagittal"])  # listed but no files on disk
        w.writerow([corrupt, "S_BAD", 0, 0, "Coronal"])
        # `empty` study: no rows at all
    return comp, good, corrupt, empty


def test_preprocess_study_missing_and_present_planes(tmp_path):
    comp, good, _, _ = _make_comp(tmp_path)
    rows = [{"SeriesInstanceUID": "S_AX", "Anatomical_Plane": "Axial"},
            {"SeriesInstanceUID": "S_MISSING_DIR", "Anatomical_Plane": "Sagittal"}]
    image, present, errors = preprocess_study(str(comp / "test_series" / good), rows)
    assert image.shape == (3,) + TARGET_SHAPE and image.dtype == np.float32
    assert present == [False, False, True]  # Sagittal, Coronal, Axial
    assert (image[0] == 0).all() and (image[1] == 0).all()
    assert len(errors) == 2  # sagittal (no files) + coronal (no series)


def test_run_inference_one_valid_row_per_study_with_fallbacks(tmp_path):
    comp, good, corrupt, empty = _make_comp(tmp_path)
    model = StubModel()
    out = tmp_path / "submission.csv"
    summary = run_inference(str(comp), model, str(out), log=lambda *a: None)

    ok, issues = validate_submission_file(str(out), [good, corrupt, empty])
    assert ok, issues
    with open(out, newline="") as f:
        rows = {r["StudyInstanceUID"]: r for r in csv.DictReader(f)}
    assert list(rows) == [good, corrupt, empty]  # test.csv order preserved
    # good study: model called once with only Axial present -> sigmoid(1)
    assert len(model.calls) == 1 and model.calls[0][1] == [[False, False, True]]
    assert float(rows[good]["ACL"]) == np.float32(torch.sigmoid(torch.tensor(1.0))).round(6)
    # corrupt series and no-series study: fallback rows, errors recorded, no crash
    for sid in (corrupt, empty):
        assert all(float(rows[sid][c]) == FALLBACK_PROB for c in LABEL_COLUMNS)
    assert summary["n_fallback"] == 2
    assert corrupt in summary["errors"] and empty in summary["errors"]


def test_nonfinite_model_output_falls_back(tmp_path):
    comp, good, _, _ = _make_comp(tmp_path)

    class NaNModel(StubModel):
        def forward(self, image, plane_present):
            return torch.full((image.shape[0], len(LABEL_COLUMNS)), float("nan"))

    out = tmp_path / "sub.csv"
    summary = run_inference(str(comp), NaNModel(), str(out), log=lambda *a: None)
    with open(out, newline="") as f:
        row = next(r for r in csv.DictReader(f) if r["StudyInstanceUID"] == good)
    assert float(row["ACL"]) == FALLBACK_PROB and summary["n_fallback"] == 3


def test_parallel_preprocessing_gives_identical_submission(tmp_path):
    comp, *_ = _make_comp(tmp_path)
    seq, par = tmp_path / "seq.csv", tmp_path / "par.csv"
    s1 = run_inference(str(comp), StubModel(), str(seq), log=lambda *a: None)
    s2 = run_inference(str(comp), StubModel(), str(par), log=lambda *a: None, n_workers=2)
    assert seq.read_text() == par.read_text()
    assert s1["n_fallback"] == s2["n_fallback"] and s1["errors"].keys() == s2["errors"].keys()
