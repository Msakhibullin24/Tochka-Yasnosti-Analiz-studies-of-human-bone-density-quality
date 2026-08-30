from __future__ import annotations

import io
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image
from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian, SecondaryCaptureImageStorage, generate_uid

from app.keypoints import KEYPOINTS, Landmark
from app.main import app
from app.settings import settings


def make_dicom(body_part: str) -> bytes:
    file_meta = FileMetaDataset()
    file_meta.MediaStorageSOPClassUID = SecondaryCaptureImageStorage
    file_meta.MediaStorageSOPInstanceUID = generate_uid()
    file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
    dataset = FileDataset(None, {}, file_meta=file_meta, preamble=b"\0" * 128)
    dataset.SOPClassUID = file_meta.MediaStorageSOPClassUID
    dataset.SOPInstanceUID = file_meta.MediaStorageSOPInstanceUID
    dataset.StudyInstanceUID = generate_uid()
    dataset.SeriesInstanceUID = generate_uid()
    dataset.PatientID = "PRIVATE-PATIENT"
    dataset.PatientIdentityRemoved = "YES"
    dataset.BodyPartExamined = body_part
    dataset.Modality = "DX"
    dataset.PhotometricInterpretation = "MONOCHROME2"
    dataset.SamplesPerPixel = 1
    dataset.Rows = 64
    dataset.Columns = 32
    dataset.BitsAllocated = 16
    dataset.BitsStored = 12
    dataset.HighBit = 11
    dataset.PixelRepresentation = 0
    pixels = np.linspace(0, 4095, dataset.Rows * dataset.Columns, dtype=np.uint16).reshape(dataset.Rows, dataset.Columns)
    dataset.PixelData = pixels.tobytes()
    output = io.BytesIO()
    dataset.save_as(output, enforce_file_format=True)
    return output.getvalue()


def make_color_dicom() -> bytes:
    file_meta = FileMetaDataset()
    file_meta.MediaStorageSOPClassUID = SecondaryCaptureImageStorage
    file_meta.MediaStorageSOPInstanceUID = generate_uid()
    file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
    dataset = FileDataset(None, {}, file_meta=file_meta, preamble=b"\0" * 128)
    dataset.SOPClassUID = file_meta.MediaStorageSOPClassUID
    dataset.SOPInstanceUID = file_meta.MediaStorageSOPInstanceUID
    dataset.StudyInstanceUID = generate_uid()
    dataset.SeriesInstanceUID = generate_uid()
    dataset.Modality = "OT"
    dataset.PhotometricInterpretation = "RGB"
    dataset.SamplesPerPixel = 3
    dataset.PlanarConfiguration = 0
    dataset.Rows = 16
    dataset.Columns = 12
    dataset.BitsAllocated = 8
    dataset.BitsStored = 8
    dataset.HighBit = 7
    dataset.PixelRepresentation = 0
    dataset.PixelData = np.zeros((16, 12, 3), dtype=np.uint8).tobytes()
    output = io.BytesIO()
    dataset.save_as(output, enforce_file_format=True)
    return output.getvalue()


def make_dataset(root: Path) -> str:
    study_id = "ST-0123456789ABCDEFFEDC"
    patient_group_id = "PG-0123456789ABCDEFFEDC"
    device_group = "DV-0123456789ABCDEFFEDC"
    (root / "processed" / study_id).mkdir(parents=True)
    (root / "raw" / study_id).mkdir(parents=True)
    Image.fromarray(np.arange(24, dtype=np.uint8).reshape(6, 4), mode="L").save(
        root / "processed" / study_id / "p_0046.png"
    )
    np.save(root / "raw" / study_id / "transmissions.npy", np.arange(144, dtype=np.uint16).reshape(6, 4, 6))
    record = {
        "schemaVersion": "1.0.0",
        "loaderVersion": "test-loader/1.0.0",
        "studyId": study_id,
        "patientGroupId": patient_group_id,
        "deviceGroup": device_group,
        "protocol": "spine_pa",
        "protocolCode": "AP Spine",
        "split": "train",
        "processedImages": [{"tag": "0x0046", "width": 4, "height": 6, "p01": 0, "p99": 255, "clippedFraction": 0}],
        "raw": {"shape": [6, 4, 6], "transmissionCount": 6, "phaseSemantics": "unverified", "saturatedFraction": 0, "phases": []},
        "quality": {"reviewRequired": False, "flags": []},
        "technicalQc": {"baselineVersion": "test", "score": 100, "trainingCandidate": True, "requiresExpertReview": False, "flags": [], "processed": [], "raw": {"saturatedFraction": 0, "phaseSemantics": "unverified"}, "limitations": []},
        "privacy": {"directIdentifiersExported": False},
    }
    (root / "manifest.jsonl").write_text(json.dumps(record) + "\n", encoding="utf-8")
    return study_id


def predictions() -> list[Landmark]:
    return [Landmark(name, 0.5, 0.5, 0.95, True) for name in KEYPOINTS]


def test_model_registry_exposes_readiness_without_overclaiming() -> None:
    response = TestClient(app).get("/api/v1/models")
    assert response.status_code == 200
    statuses = {item["protocol"]: item["status"] for item in response.json()["models"]}
    assert statuses == {"total-body": "ready", "spine": "planned", "hip": "planned"}


def test_total_body_dicom_returns_research_landmarks(monkeypatch) -> None:
    monkeypatch.setattr("app.service.model.predict", lambda image: predictions())
    response = TestClient(app).post(
        "/api/v1/studies/analyze",
        files={"file": ("total-body.dcm", make_dicom("WHOLE BODY"), "application/dicom")},
    )
    assert response.status_code == 200
    study = response.json()
    assert study["type"] == "total-body"
    assert study["provenance"]["mode"] == "research-model"
    assert study["routing"]["modelStatus"] == "ready"
    assert study["routing"]["protocol"] == "total-body"
    assert len(study["landmarks"]) == 105
    assert "PRIVATE-PATIENT" not in response.text


def test_spine_is_rejected_by_total_body_router() -> None:
    response = TestClient(app).post(
        "/api/v1/studies/analyze",
        files={"file": ("spine.dcm", make_dicom("LSPINE"), "application/dicom")},
    )
    assert response.status_code == 422
    assert response.json()["detail"]["code"] == "UNSUPPORTED_PROTOCOL"
    assert response.json()["detail"]["routing"]["modelStatus"] == "planned"


def test_invalid_manual_override_is_rejected() -> None:
    response = TestClient(app).post(
        "/api/v1/studies/analyze",
        files={"file": ("study.dcm", make_dicom("WHOLE BODY"), "application/dicom")},
        data={"protocol_override": "brain"},
    )
    assert response.status_code == 422
    assert response.json()["detail"]["code"] == "INVALID_PROTOCOL_OVERRIDE"


def test_color_secondary_capture_is_excluded_and_audited(tmp_path: Path, monkeypatch) -> None:
    exclusion_log = tmp_path / "exclusions.jsonl"
    monkeypatch.setattr(
        "app.main.settings",
        replace(settings, exclusion_log=exclusion_log, audit_log=tmp_path / "audit.jsonl"),
    )
    client = TestClient(app)
    response = client.post(
        "/api/v1/studies/analyze",
        files={"file": ("print.dcm", make_color_dicom(), "application/dicom")},
    )
    assert response.status_code == 422
    detail = response.json()["detail"]
    assert detail["code"] == "SECONDARY_CAPTURE_EXCLUDED"
    assert detail["trainingEligible"] is False
    assert detail["exclusionId"].startswith("EX-")
    registry = client.get("/api/v1/exclusions").json()
    assert registry["count"] == 1
    assert registry["exclusions"][0]["objectId"] == detail["exclusionId"]
    assert "print.dcm" not in exclusion_log.read_text(encoding="utf-8")


def test_dataset_workbench_api_annotation_and_exports(tmp_path: Path, monkeypatch) -> None:
    study_id = make_dataset(tmp_path)
    annotation_root = tmp_path / "annotations"
    monkeypatch.setattr(
        "app.main.settings",
        replace(settings, dataset_root=tmp_path, annotation_root=annotation_root, longitudinal_root=tmp_path / "longitudinal", audit_log=tmp_path / "audit.jsonl"),
    )
    client = TestClient(app)
    summary_response = client.get("/api/v1/datasets/current", headers={"X-Request-ID": "test-request-0001"})
    assert summary_response.json()["studyCount"] == 1
    assert summary_response.headers["X-Request-ID"] == "test-request-0001"
    assert summary_response.headers["Cache-Control"] == "no-store"
    integrity = client.get("/api/v1/datasets/current/integrity").json()
    assert integrity["status"] == "pass"
    assert integrity["datasetVersion"].startswith("DS-")
    readiness = client.get("/api/v1/readiness").json()
    assert readiness["stage"] == "research"
    assert readiness["criticalBlockerCount"] > 0
    assert next(gate for gate in readiness["gates"] if gate["id"] == "access_control")["status"] == "block"
    study = client.get(f"/api/v1/datasets/current/studies/{study_id}").json()
    assert study["assets"][0]["name"] == "p_0046.png"
    assert client.get(f"/api/v1/datasets/current/studies/{study_id}/raw/0.png").headers["content-type"] == "image/png"

    annotation = {
        "schemaVersion": "1.0.0",
        "studyId": study_id,
        "patientGroupId": study["patientGroupId"],
        "deviceGroup": study["deviceGroup"],
        "protocol": "spine_pa",
        "evaluable": True,
        "notEvaluableReason": "",
        "overallAction": "accept",
        "defects": [{"code": "spine_tilt", "present": False, "severity": "none"}],
        "landmarks": [{"name": "L1", "x": 0.5, "y": 0.25, "visible": True}],
        "regions": [{"name": "L1_roi", "geometryType": "box", "points": [[0.2, 0.1], [0.7, 0.4]]}],
        "expert": {"readerId": "reader-01", "readIndex": 1, "confidence": "high", "createdAt": "2026-08-30T12:00:00Z", "adjudicated": False, "comment": ""},
    }
    saved = client.put(f"/api/v1/datasets/current/studies/{study_id}/annotations", json=annotation)
    assert saved.status_code == 200
    assert len(list(annotation_root.glob(f"{study_id}--*.json"))) == 1
    coco = client.get("/api/v1/datasets/current/exports/coco")
    assert coco.status_code == 200
    assert coco.json()["annotations"][0]["bbox"] == pytest.approx([0.8, 0.6, 2.0, 1.8])
    jsonl = client.get("/api/v1/datasets/current/exports/annotations")
    assert jsonl.status_code == 200
    assert json.loads(jsonl.text)["expert"]["readerId"] == "reader-01"
    assert client.get("/api/v1/audit/status").json()["eventCount"] == 3

    second = {
        **annotation,
        "overallAction": "repeat",
        "defects": [{"code": "spine_tilt", "present": True, "severity": "major", "action": "repeat"}],
        "expert": {**annotation["expert"], "readerId": "reader-02", "createdAt": "2026-08-30T12:30:00Z"},
    }
    assert client.put(f"/api/v1/datasets/current/studies/{study_id}/annotations", json=second).status_code == 200
    agreement = client.get("/api/v1/datasets/current/agreement").json()
    assert agreement["doubleReadStudyCount"] == 1
    assert agreement["pendingAdjudicationCount"] == 1
    assert agreement["overallAction"]["percentAgreement"] == 0
    assert client.get("/api/v1/datasets/current/adjudication").json()["count"] == 1

    adjudication = {
        **annotation,
        "overallAction": "review",
        "expert": {
            **annotation["expert"],
            "readerId": "reader-03",
            "createdAt": "2026-08-30T13:00:00Z",
            "adjudicated": True,
        },
    }
    assert client.put(f"/api/v1/datasets/current/studies/{study_id}/annotations", json=adjudication).status_code == 200
    resolved = client.get("/api/v1/datasets/current/agreement").json()
    assert resolved["adjudicatedStudyCount"] == 1
    assert resolved["pendingAdjudicationCount"] == 0

    lsc_profile = {
        "schemaVersion": "2.0.0",
        "profileId": "LSC-TEST-SPINE",
        "version": "1.0.0",
        "facilityId": "FACILITY-TEST",
        "deviceGroup": study["deviceGroup"],
        "operatorGroup": "OPERATORS-TEST",
        "protocol": "spine_pa",
        "validFrom": "2025-01-01",
        "validTo": "2027-12-31",
        "sites": [{"site": "l1-l4", "percent": 5.3}],
        "precisionStudyReference": "precision-test-01",
        "approvedBy": "physicist-01",
        "status": "active",
    }
    assert client.put("/api/v1/longitudinal/lsc-profiles/LSC-TEST-SPINE", json=lsc_profile).status_code == 200
    measurement = {
        "schemaVersion": "2.0.0",
        "studyId": study_id,
        "patientGroupId": study["patientGroupId"],
        "acquiredOn": "2026-08-30",
        "protocol": "spine_pa",
        "deviceGroup": study["deviceGroup"],
        "facilityId": "FACILITY-TEST",
        "operatorGroup": "OPERATORS-TEST",
        "qualityStatus": "accept",
        "positioningStatus": "accept",
        "roiStatus": "accept",
        "source": "manual-verified",
        "sourceReference": "vendor-report-test-01",
        "expertConfirmed": True,
        "confirmedBy": "reader-01",
        "sites": [{"site": "l1-l4", "bmd": 0.842}],
    }
    measurement_response = client.put(f"/api/v1/longitudinal/measurements/{study_id}", json=measurement)
    assert measurement_response.status_code == 200
    assert measurement_response.json()["confirmedBy"].startswith("ACT-")
    timeline = client.get(f"/api/v1/longitudinal/patients/{study['patientGroupId']}/timeline").json()
    assert timeline["count"] == 1
    assert client.get(f"/api/v1/longitudinal/measurements/{study_id}/baseline-candidates").json()["count"] == 0
    assert client.get("/api/v1/longitudinal/comparisons/CMP-00000000000000000000").status_code == 404
    assert client.get("/api/v1/longitudinal/comparisons/not-safe").status_code == 400
