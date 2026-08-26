from __future__ import annotations

import io

import numpy as np
from fastapi.testclient import TestClient
from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian, SecondaryCaptureImageStorage, generate_uid

from app.keypoints import KEYPOINTS, Landmark
from app.main import app


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
