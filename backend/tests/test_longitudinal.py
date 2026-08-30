from __future__ import annotations

import json
from pathlib import Path

from jsonschema import Draft202012Validator, FormatChecker

from app.dataset import DatasetRepository
from app.longitudinal import (
    ComparisonRequest,
    CrossCalibration,
    LongitudinalMeasurement,
    LongitudinalRepository,
    LscProfile,
    baseline_candidates,
    compare_measurements,
)


BASELINE_ID = "ST-11111111111111111111"
CURRENT_ID = "ST-22222222222222222222"
PATIENT_ID = "PG-AAAAAAAAAAAAAAAAAAAA"
DEVICE_A = "DV-AAAAAAAAAAAAAAAAAAAA"
DEVICE_B = "DV-BBBBBBBBBBBBBBBBBBBB"


def make_repository(tmp_path: Path, *, current_device: str = DEVICE_A) -> LongitudinalRepository:
    rows = [
        {"studyId": BASELINE_ID, "patientGroupId": PATIENT_ID, "deviceGroup": DEVICE_A, "protocol": "spine_pa", "privacy": {"directIdentifiersExported": False}},
        {"studyId": CURRENT_ID, "patientGroupId": PATIENT_ID, "deviceGroup": current_device, "protocol": "spine_pa", "privacy": {"directIdentifiersExported": False}},
    ]
    (tmp_path / "dataset").mkdir()
    (tmp_path / "dataset" / "manifest.jsonl").write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    return LongitudinalRepository(tmp_path / "registry", DatasetRepository(tmp_path / "dataset"))


def measurement(study_id: str, acquired_on: str, bmd: float, device: str = DEVICE_A, **patch) -> LongitudinalMeasurement:
    values = {
        "studyId": study_id,
        "patientGroupId": PATIENT_ID,
        "acquiredOn": acquired_on,
        "protocol": "spine_pa",
        "deviceGroup": device,
        "facilityId": "FACILITY-01",
        "operatorGroup": "OPERATORS-01",
        "qualityStatus": "accept",
        "positioningStatus": "accept",
        "roiStatus": "accept",
        "source": "manual-verified",
        "sourceReference": "vendor-report-sha256:012345",
        "expertConfirmed": True,
        "confirmedBy": "reader-01",
        "sites": [{"site": "l1-l4", "bmd": bmd}],
    }
    return LongitudinalMeasurement.model_validate({**values, **patch})


def profile(device: str = DEVICE_A) -> LscProfile:
    return LscProfile.model_validate({
        "profileId": "LSC-FACILITY01-SPINE",
        "version": "1.0.0",
        "facilityId": "FACILITY-01",
        "deviceGroup": device,
        "operatorGroup": "OPERATORS-01",
        "protocol": "spine_pa",
        "validFrom": "2024-01-01",
        "validTo": "2027-12-31",
        "sites": [{"site": "l1-l4", "percent": 5.3}],
        "precisionStudyReference": "precision-study-2024-01",
        "approvedBy": "medical-physicist-01",
    })


def test_comparison_uses_verified_measurements_and_facility_lsc(tmp_path: Path) -> None:
    repository = make_repository(tmp_path)
    repository.save_measurement(measurement(BASELINE_ID, "2024-08-20", 0.842))
    repository.save_measurement(measurement(CURRENT_ID, "2026-08-25", 0.891))
    repository.save_lsc_profile(profile())
    result = compare_measurements(repository, ComparisonRequest(
        baselineStudyId=BASELINE_ID,
        currentStudyId=CURRENT_ID,
        lscProfileId="LSC-FACILITY01-SPINE",
    ))
    assert result["status"] == "comparable"
    assert result["assumed"] is False
    assert result["sites"][0]["absoluteChange"] == 0.049
    assert result["sites"][0]["percentChange"] == 5.8
    assert result["sites"][0]["status"] == "significant-gain"
    assert (repository.root / "comparisons" / f'{result["comparisonId"]}.json').is_file()
    assert compare_measurements(repository, ComparisonRequest(
        baselineStudyId=BASELINE_ID,
        currentStudyId=CURRENT_ID,
        lscProfileId="LSC-FACILITY01-SPINE",
    ))["comparedAt"] == result["comparedAt"]
    assert baseline_candidates(repository, CURRENT_ID)["candidates"][0]["eligible"] is True
    schema = json.loads((Path(__file__).parents[2] / "docs" / "longitudinal.schema.json").read_text(encoding="utf-8"))
    Draft202012Validator(schema, format_checker=FormatChecker()).validate(result)


def test_registry_preserves_measurement_history_and_immutable_evidence(tmp_path: Path) -> None:
    repository = make_repository(tmp_path)
    repository.save_measurement(measurement(BASELINE_ID, "2024-08-20", 0.842))
    repository.save_measurement(measurement(BASELINE_ID, "2024-08-20", 0.843))
    history = list((repository.root / "measurement-history" / BASELINE_ID).glob("REV-*.json"))
    assert len(history) == 1
    assert json.loads(history[0].read_text(encoding="utf-8"))["sites"][0]["bmd"] == 0.842

    initial = profile()
    repository.save_lsc_profile(initial)
    assert repository.save_lsc_profile(initial)["profileId"] == initial.profileId
    changed = LscProfile.model_validate({**initial.model_dump(mode="json"), "sites": [{"site": "l1-l4", "percent": 4.9}]})
    try:
        repository.save_lsc_profile(changed)
    except Exception as error:
        assert "immutable" in str(error)
    else:
        raise AssertionError("An existing LSC evidence ID must not be mutable")


def test_reviewable_roi_never_produces_interpreted_trend(tmp_path: Path) -> None:
    repository = make_repository(tmp_path)
    repository.save_measurement(measurement(BASELINE_ID, "2024-08-20", 0.842))
    repository.save_measurement(measurement(CURRENT_ID, "2026-08-25", 0.891, roiStatus="review"))
    repository.save_lsc_profile(profile())
    result = compare_measurements(repository, ComparisonRequest(
        baselineStudyId=BASELINE_ID,
        currentStudyId=CURRENT_ID,
        lscProfileId="LSC-FACILITY01-SPINE",
    ))
    assert result["status"] == "review"
    assert result["sites"][0]["status"] == "not-comparable"


def test_device_change_is_blocked_without_valid_cross_calibration(tmp_path: Path) -> None:
    repository = make_repository(tmp_path, current_device=DEVICE_B)
    repository.save_measurement(measurement(BASELINE_ID, "2024-08-20", 0.842))
    repository.save_measurement(measurement(CURRENT_ID, "2026-08-25", 0.891, device=DEVICE_B))
    repository.save_lsc_profile(profile(DEVICE_B))
    request = ComparisonRequest(
        baselineStudyId=BASELINE_ID,
        currentStudyId=CURRENT_ID,
        lscProfileId="LSC-FACILITY01-SPINE",
    )
    assert compare_measurements(repository, request)["status"] == "not-comparable"

    repository.save_cross_calibration(CrossCalibration.model_validate({
        "calibrationId": "XCL-FACILITY01-A-B",
        "facilityId": "FACILITY-01",
        "fromDeviceGroup": DEVICE_A,
        "toDeviceGroup": DEVICE_B,
        "validFrom": "2026-01-01",
        "validTo": "2027-12-31",
        "sites": ["l1-l4"],
        "evidenceReference": "cross-calibration-report-2026",
        "approvedBy": "medical-physicist-01",
    }))
    calibrated = compare_measurements(repository, request.model_copy(update={"crossCalibrationId": "XCL-FACILITY01-A-B"}))
    assert calibrated["status"] == "comparable"
