import warnings

import pytest

from conftest import synthetic_spine, write_dicom
from dxaqc.dynamics import compare
from dxaqc.geometry import measure_spine

warnings.filterwarnings("ignore")


def _visit(angle, quality=0, region="spine"):
    return {"region": region, "quality": quality, "violations": ["spine_axis"] if quality else [],
            "features": measure_spine(synthetic_spine(angle)).features}


def test_same_positioning_is_comparable_and_small_change_is_not_significant():
    r = compare(_visit(1.0), _visit(1.5), baseline_bmd=0.900, followup_bmd=0.890, precision_cv_percent=1.0)
    assert r.verdict == "comparable"
    assert r.bmd["lsc_percent"] == 2.77 and "незначимо" in r.bmd["interpretation"]


def test_change_above_lsc_is_significant_only_when_scans_are_comparable():
    ok = compare(_visit(0.5), _visit(1.0), baseline_bmd=0.900, followup_bmd=0.840, lsc_percent=3.0)
    assert ok.verdict == "comparable" and "значимо" in ok.bmd["interpretation"] and "незначимо" not in ok.bmd["interpretation"]
    tilted = compare(_visit(0.5), _visit(8.0, quality=1), baseline_bmd=0.900, followup_bmd=0.840, lsc_percent=3.0)
    assert tilted.verdict == "not_comparable" and "не интерпретируется" in tilted.bmd["interpretation"]
    assert any(c.id == "spine_angle_deg" and c.status == "block" for c in tilted.checks)


def test_different_region_or_side_blocks_the_comparison():
    assert compare(_visit(0), {**_visit(0), "region": "hip_left"}).verdict == "not_comparable"
    left, right = {**_visit(0), "region": "hip_left"}, {**_visit(0), "region": "hip_right"}
    r = compare(left, right)
    assert r.verdict == "not_comparable" and any(c.id == "side" and c.status == "block" for c in r.checks)


def test_bmd_is_never_invented_and_result_declares_unvalidated_tolerances():
    r = compare(_visit(0), _visit(0.3))
    assert r.bmd is None and r.to_dict()["tolerances_clinically_validated"] is False


def test_compare_endpoint(tmp_path, bundle_available, monkeypatch):
    monkeypatch.setenv("DXAQC_DATA_DIR", str(tmp_path / "jobs"))
    import importlib
    import dxaqc.api as api
    importlib.reload(api)
    from fastapi.testclient import TestClient
    client = TestClient(api.app)
    a = write_dicom(tmp_path / "a.dcm", synthetic_spine(0), sop_uid="9.1")
    b = write_dicom(tmp_path / "b.dcm", synthetic_spine(0), sop_uid="9.2")
    resp = client.post("/api/v1/compare", files={"baseline": ("a.dcm", a.read_bytes()), "followup": ("b.dcm", b.read_bytes())},
                       data={"baseline_bmd": "0.9", "followup_bmd": "0.85", "lsc_percent": "3"})
    body = resp.json()
    assert resp.status_code == 200 and body["processing_status"] == "Success"
    assert body["verdict"] in {"comparable", "review"} and body["bmd_change"]["change_percent"] == pytest.approx(-5.56, abs=0.01)
    bad = client.post("/api/v1/compare", files={"baseline": ("a.dcm", b"junk"), "followup": ("b.dcm", b.read_bytes())})
    assert bad.status_code == 422
