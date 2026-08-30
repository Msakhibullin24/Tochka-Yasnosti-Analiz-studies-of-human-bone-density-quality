from __future__ import annotations

import hashlib
import json
import os
import tempfile
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .dataset import DatasetContractError, DatasetRepository


ENGINE_VERSION = "osseo-longitudinal/2.0.0"
Protocol = Literal["spine_pa", "hip_left", "hip_right"]
MeasurementSite = Literal["l1-l4", "total-hip", "femoral-neck"]
ReviewStatus = Literal["accept", "review", "reject"]
SAFE_ID = r"^[A-Z][A-Z0-9._-]{2,79}$"


class SiteMeasurement(BaseModel):
    model_config = ConfigDict(extra="forbid")

    site: MeasurementSite
    bmd: float = Field(gt=0, le=5)


class LongitudinalMeasurement(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schemaVersion: Literal["2.0.0"] = "2.0.0"
    studyId: str = Field(pattern=r"^ST-[A-F0-9]{20}$")
    patientGroupId: str = Field(pattern=r"^PG-[A-F0-9]{20}$")
    acquiredOn: date
    protocol: Protocol
    deviceGroup: str = Field(pattern=r"^DV-[A-F0-9]{20}$")
    facilityId: str = Field(pattern=SAFE_ID)
    operatorGroup: str | None = Field(default=None, pattern=SAFE_ID)
    qualityStatus: ReviewStatus
    positioningStatus: ReviewStatus
    roiStatus: ReviewStatus
    source: Literal["vendor-structured", "dicom-sr", "manual-verified"]
    sourceReference: str = Field(min_length=3, max_length=160)
    expertConfirmed: bool
    confirmedBy: str = Field(min_length=2, max_length=120)
    sites: list[SiteMeasurement] = Field(min_length=1, max_length=3)
    recordedAt: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    @model_validator(mode="after")
    def validate_sites_and_confirmation(self) -> "LongitudinalMeasurement":
        site_names = [item.site for item in self.sites]
        if len(site_names) != len(set(site_names)):
            raise ValueError("Measurement sites must be unique")
        allowed = {"l1-l4"} if self.protocol == "spine_pa" else {"total-hip", "femoral-neck"}
        if not set(site_names).issubset(allowed):
            raise ValueError("Measurement site does not match the protocol")
        if not self.expertConfirmed:
            raise ValueError("Structured BMD must be confirmed before longitudinal use")
        return self


class LscSite(BaseModel):
    model_config = ConfigDict(extra="forbid")

    site: MeasurementSite
    percent: float = Field(ge=0.1, le=25)


class LscProfile(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schemaVersion: Literal["2.0.0"] = "2.0.0"
    profileId: str = Field(pattern=r"^LSC-[A-Z0-9._-]{4,76}$")
    version: str = Field(pattern=r"^[0-9]+\.[0-9]+\.[0-9]+$")
    facilityId: str = Field(pattern=SAFE_ID)
    deviceGroup: str = Field(pattern=r"^DV-[A-F0-9]{20}$")
    operatorGroup: str | None = Field(default=None, pattern=SAFE_ID)
    protocol: Protocol
    validFrom: date
    validTo: date
    sites: list[LscSite] = Field(min_length=1, max_length=3)
    precisionStudyReference: str = Field(min_length=3, max_length=160)
    approvedBy: str = Field(min_length=2, max_length=120)
    status: Literal["active", "retired"] = "active"
    recordedAt: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    @model_validator(mode="after")
    def validate_profile(self) -> "LscProfile":
        if self.validTo < self.validFrom:
            raise ValueError("LSC validity range is inverted")
        names = [item.site for item in self.sites]
        if len(names) != len(set(names)):
            raise ValueError("LSC sites must be unique")
        allowed = {"l1-l4"} if self.protocol == "spine_pa" else {"total-hip", "femoral-neck"}
        if not set(names).issubset(allowed):
            raise ValueError("LSC site does not match the protocol")
        return self


class CrossCalibration(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schemaVersion: Literal["2.0.0"] = "2.0.0"
    calibrationId: str = Field(pattern=r"^XCL-[A-Z0-9._-]{4,76}$")
    facilityId: str = Field(pattern=SAFE_ID)
    fromDeviceGroup: str = Field(pattern=r"^DV-[A-F0-9]{20}$")
    toDeviceGroup: str = Field(pattern=r"^DV-[A-F0-9]{20}$")
    validFrom: date
    validTo: date
    sites: list[MeasurementSite] = Field(min_length=1, max_length=3)
    evidenceReference: str = Field(min_length=3, max_length=160)
    approvedBy: str = Field(min_length=2, max_length=120)
    status: Literal["active", "retired"] = "active"
    recordedAt: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    @model_validator(mode="after")
    def validate_calibration(self) -> "CrossCalibration":
        if self.fromDeviceGroup == self.toDeviceGroup:
            raise ValueError("Cross-calibration requires two different devices")
        if self.validTo < self.validFrom:
            raise ValueError("Cross-calibration validity range is inverted")
        if len(self.sites) != len(set(self.sites)):
            raise ValueError("Cross-calibration sites must be unique")
        return self


class ComparisonRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    baselineStudyId: str = Field(pattern=r"^ST-[A-F0-9]{20}$")
    currentStudyId: str = Field(pattern=r"^ST-[A-F0-9]{20}$")
    lscProfileId: str = Field(pattern=r"^LSC-[A-Z0-9._-]{4,76}$")
    crossCalibrationId: str | None = Field(default=None, pattern=r"^XCL-[A-Z0-9._-]{4,76}$")

    @model_validator(mode="after")
    def different_studies(self) -> "ComparisonRequest":
        if self.baselineStudyId == self.currentStudyId:
            raise ValueError("Baseline and current study must differ")
        return self


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    encoded = json.dumps(value, ensure_ascii=False, indent=2) + "\n"
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as stream:
        stream.write(encoded)
        temporary = Path(stream.name)
    temporary.chmod(0o600)
    os.replace(temporary, path)


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise KeyError(path.stem)
    return json.loads(path.read_text(encoding="utf-8"))


def _semantic_payload(value: dict[str, Any]) -> dict[str, Any]:
    """Remove the server timestamp when checking an idempotent registry write."""
    return {key: item for key, item in value.items() if key != "recordedAt"}


class LongitudinalRepository:
    def __init__(self, root: Path, dataset: DatasetRepository):
        self.root = root.expanduser().resolve()
        self.dataset = dataset

    def _path(self, collection: str, identifier: str) -> Path:
        if not identifier or any(character not in "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-._" for character in identifier):
            raise ValueError("Unsafe longitudinal registry identifier")
        return self.root / collection / f"{identifier}.json"

    def save_measurement(self, value: LongitudinalMeasurement) -> dict[str, Any]:
        study = self.dataset.get_study(value.studyId)
        if study.get("patientGroupId") != value.patientGroupId:
            raise DatasetContractError("Measurement patientGroupId does not match the dataset")
        if study.get("deviceGroup") != value.deviceGroup:
            raise DatasetContractError("Measurement deviceGroup does not match the dataset")
        if study.get("protocol") != value.protocol:
            raise DatasetContractError("Measurement protocol does not match the dataset")
        payload = value.model_dump(mode="json")
        path = self._path("measurements", value.studyId)
        if path.is_file():
            previous = _read_json(path)
            revision = hashlib.sha256(json.dumps(previous, sort_keys=True).encode("utf-8")).hexdigest()[:20].upper()
            history_path = self.root / "measurement-history" / value.studyId / f"REV-{revision}.json"
            if not history_path.is_file():
                _atomic_json(history_path, previous)
        _atomic_json(path, payload)
        return payload

    def measurement(self, study_id: str) -> LongitudinalMeasurement:
        return LongitudinalMeasurement.model_validate(_read_json(self._path("measurements", study_id)))

    def measurements(self, patient_group_id: str | None = None) -> list[LongitudinalMeasurement]:
        values: list[LongitudinalMeasurement] = []
        directory = self.root / "measurements"
        for path in sorted(directory.glob("*.json")) if directory.is_dir() else []:
            value = LongitudinalMeasurement.model_validate(_read_json(path))
            if patient_group_id is None or value.patientGroupId == patient_group_id:
                values.append(value)
        return sorted(values, key=lambda item: (item.acquiredOn, item.studyId))

    def save_lsc_profile(self, value: LscProfile) -> dict[str, Any]:
        payload = value.model_dump(mode="json")
        path = self._path("lsc-profiles", value.profileId)
        if path.is_file():
            existing = _read_json(path)
            if _semantic_payload(existing) == _semantic_payload(payload):
                return existing
            raise DatasetContractError("LSC profile IDs are immutable; create a new profileId and version")
        _atomic_json(path, payload)
        return payload

    def lsc_profile(self, profile_id: str) -> LscProfile:
        return LscProfile.model_validate(_read_json(self._path("lsc-profiles", profile_id)))

    def lsc_profiles(self) -> list[LscProfile]:
        directory = self.root / "lsc-profiles"
        return [LscProfile.model_validate(_read_json(path)) for path in sorted(directory.glob("*.json"))] if directory.is_dir() else []

    def save_cross_calibration(self, value: CrossCalibration) -> dict[str, Any]:
        payload = value.model_dump(mode="json")
        path = self._path("cross-calibrations", value.calibrationId)
        if path.is_file():
            existing = _read_json(path)
            if _semantic_payload(existing) == _semantic_payload(payload):
                return existing
            raise DatasetContractError("Cross-calibration IDs are immutable; create a new calibrationId")
        _atomic_json(path, payload)
        return payload

    def cross_calibration(self, calibration_id: str) -> CrossCalibration:
        return CrossCalibration.model_validate(_read_json(self._path("cross-calibrations", calibration_id)))

    def cross_calibrations(self) -> list[CrossCalibration]:
        directory = self.root / "cross-calibrations"
        return [CrossCalibration.model_validate(_read_json(path)) for path in sorted(directory.glob("*.json"))] if directory.is_dir() else []

    def comparison(self, comparison_id: str) -> dict[str, Any] | None:
        path = self._path("comparisons", comparison_id)
        return _read_json(path) if path.is_file() else None

    def save_comparison(self, value: dict[str, Any]) -> dict[str, Any]:
        path = self._path("comparisons", str(value["comparisonId"]))
        if path.is_file():
            return _read_json(path)
        _atomic_json(path, value)
        return value


def _check(identifier: str, label: str, status: Literal["pass", "review", "block"], detail: str, *, critical: bool = True) -> dict[str, Any]:
    return {"id": identifier, "label": label, "status": status, "passed": status == "pass", "critical": critical, "detail": detail}


def _change(baseline: float, current: float, lsc_percent: float, comparable: bool) -> tuple[float, float, str]:
    absolute = round(current - baseline, 3)
    percent = round((current - baseline) / baseline * 100, 1)
    if not comparable:
        status = "not-comparable"
    elif percent >= lsc_percent:
        status = "significant-gain"
    elif percent <= -lsc_percent:
        status = "significant-loss"
    else:
        status = "stable"
    return absolute, percent, status


def compare_measurements(repository: LongitudinalRepository, request: ComparisonRequest) -> dict[str, Any]:
    baseline = repository.measurement(request.baselineStudyId)
    current = repository.measurement(request.currentStudyId)
    profile = repository.lsc_profile(request.lscProfileId)
    baseline_sites = {item.site: item.bmd for item in baseline.sites}
    current_sites = {item.site: item.bmd for item in current.sites}
    shared_sites = sorted(set(baseline_sites) & set(current_sites))
    profile_sites = {item.site: item.percent for item in profile.sites}
    checks: list[dict[str, Any]] = []

    checks.append(_check("patient", "Один пациент", "pass" if baseline.patientGroupId == current.patientGroupId else "block", "patientGroupId совпадает" if baseline.patientGroupId == current.patientGroupId else "Исследования относятся к разным patientGroupId"))
    checks.append(_check("chronology", "Хронология", "pass" if baseline.acquiredOn < current.acquiredOn else "block", f"{baseline.acquiredOn.isoformat()} → {current.acquiredOn.isoformat()}"))
    checks.append(_check("protocol", "Протокол и сторона", "pass" if baseline.protocol == current.protocol else "block", f"{baseline.protocol} → {current.protocol}"))
    structured_ok = baseline.expertConfirmed and current.expertConfirmed and bool(shared_sites)
    checks.append(_check("structured_bmd", "Подтверждённые BMD", "pass" if structured_ok else "block", f"общие ROI={','.join(shared_sites) or 'нет'}"))

    quality_values = {baseline.qualityStatus, current.qualityStatus}
    quality_state: Literal["pass", "review", "block"] = "block" if "reject" in quality_values else "review" if "review" in quality_values else "pass"
    checks.append(_check("quality", "Качество исследований", quality_state, f"baseline={baseline.qualityStatus}; current={current.qualityStatus}"))
    positioning_values = {baseline.positioningStatus, current.positioningStatus}
    positioning_state: Literal["pass", "review", "block"] = "block" if "reject" in positioning_values else "review" if "review" in positioning_values else "pass"
    checks.append(_check("positioning", "Сопоставимая укладка", positioning_state, f"baseline={baseline.positioningStatus}; current={current.positioningStatus}", critical=False))
    roi_values = {baseline.roiStatus, current.roiStatus}
    roi_state: Literal["pass", "review", "block"] = "block" if "reject" in roi_values else "review" if "review" in roi_values else "pass"
    checks.append(_check("roi", "Сопоставимые ROI", roi_state, f"baseline={baseline.roiStatus}; current={current.roiStatus}", critical=False))

    same_device = baseline.deviceGroup == current.deviceGroup
    checks.append(_check("device", "Аппарат", "pass", "Аппарат не менялся" if same_device else "Аппарат изменился; решение определяется cross-calibration gate", critical=False))
    calibration: CrossCalibration | None = None
    calibration_ok = same_device
    calibration_detail = "Не требуется: аппарат не менялся"
    if not same_device:
        if request.crossCalibrationId:
            calibration = repository.cross_calibration(request.crossCalibrationId)
            device_pair = {calibration.fromDeviceGroup, calibration.toDeviceGroup} == {baseline.deviceGroup, current.deviceGroup}
            calibration_ok = (
                calibration.status == "active"
                and calibration.facilityId == current.facilityId
                and device_pair
                and calibration.validFrom <= current.acquiredOn <= calibration.validTo
                and set(shared_sites).issubset(set(calibration.sites))
            )
            calibration_detail = f"{calibration.calibrationId}; valid={calibration_ok}"
        else:
            calibration_detail = "Запись cross-calibration не указана"
    checks.append(_check("cross_calibration", "Cross-calibration", "pass" if calibration_ok else "block", calibration_detail))

    profile_ok = (
        profile.status == "active"
        and profile.facilityId == current.facilityId
        and profile.deviceGroup == current.deviceGroup
        and profile.protocol == current.protocol
        and profile.validFrom <= current.acquiredOn <= profile.validTo
        and (profile.operatorGroup is None or profile.operatorGroup == current.operatorGroup)
        and bool(shared_sites)
        and set(shared_sites).issubset(profile_sites)
    )
    checks.append(_check("lsc_profile", "Профиль LSC", "pass" if profile_ok else "block", f"{profile.profileId}@{profile.version}; valid={profile_ok}"))

    if any(item["status"] == "block" for item in checks):
        comparison_status = "not-comparable"
    elif any(item["status"] == "review" for item in checks):
        comparison_status = "review"
    else:
        comparison_status = "comparable"
    sites: list[dict[str, Any]] = []
    for site in shared_sites:
        lsc_percent = profile_sites.get(site)
        if lsc_percent is None:
            continue
        absolute, percent, status = _change(baseline_sites[site], current_sites[site], lsc_percent, comparison_status == "comparable")
        sites.append({
            "site": site,
            "baselineBmd": baseline_sites[site],
            "currentBmd": current_sites[site],
            "absoluteChange": absolute,
            "percentChange": percent,
            "lscPercent": lsc_percent,
            "status": status,
        })
    identity = json.dumps({"request": request.model_dump(mode="json"), "baselineRecordedAt": baseline.recordedAt.isoformat(), "currentRecordedAt": current.recordedAt.isoformat(), "profileVersion": profile.version}, sort_keys=True)
    comparison_id = "CMP-" + hashlib.sha256(identity.encode("utf-8")).hexdigest()[:20].upper()
    stored = repository.comparison(comparison_id)
    if stored is not None:
        return stored
    result = {
        "schemaVersion": "2.0.0",
        "comparisonId": comparison_id,
        "engineVersion": ENGINE_VERSION,
        "decisionBasis": "deterministic-quality-gates",
        "baselineStudyId": baseline.studyId,
        "currentStudyId": current.studyId,
        "patientGroupId": current.patientGroupId,
        "baselineDate": baseline.acquiredOn.isoformat(),
        "currentDate": current.acquiredOn.isoformat(),
        "intervalMonths": (current.acquiredOn.year - baseline.acquiredOn.year) * 12 + current.acquiredOn.month - baseline.acquiredOn.month,
        "lscProfileId": profile.profileId,
        "lscProfileVersion": profile.version,
        "crossCalibrationId": calibration.calibrationId if calibration else None,
        "status": comparison_status,
        "assumed": False,
        "checks": checks,
        "sites": sites,
        "recommendationCode": "INTERPRET_WITH_CLINICAL_CONTEXT" if comparison_status == "comparable" else "EXPERT_REVIEW_REQUIRED" if comparison_status == "review" else "COMPARISON_BLOCKED",
        "comparedAt": datetime.now(timezone.utc).isoformat(),
    }
    return repository.save_comparison(result)


def baseline_candidates(repository: LongitudinalRepository, current_study_id: str) -> dict[str, Any]:
    current = repository.measurement(current_study_id)
    candidates = []
    for value in reversed(repository.measurements(current.patientGroupId)):
        if value.studyId == current.studyId or value.acquiredOn >= current.acquiredOn:
            continue
        reasons: list[str] = []
        if value.protocol != current.protocol:
            reasons.append("protocol_mismatch")
        if value.qualityStatus == "reject":
            reasons.append("baseline_quality_rejected")
        candidates.append({
            "studyId": value.studyId,
            "acquiredOn": value.acquiredOn.isoformat(),
            "protocol": value.protocol,
            "deviceGroup": value.deviceGroup,
            "eligible": not reasons,
            "reasons": reasons,
        })
    return {"currentStudyId": current.studyId, "patientGroupId": current.patientGroupId, "candidates": candidates, "count": len(candidates)}
