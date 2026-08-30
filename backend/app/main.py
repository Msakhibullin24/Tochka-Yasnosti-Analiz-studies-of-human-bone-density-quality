from __future__ import annotations

import json
import re
import uuid

from fastapi import FastAPI, File, Form, HTTPException, Query, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response

from .annotations import AnnotationDocument
from .audit import append_audit, pseudonymous_actor, verify_audit
from .dataset import DatasetContractError, DatasetRepository, DatasetUnavailable
from .evidence import annotation_agreement, dataset_integrity, readiness_report
from .exclusions import read_exclusions, record_exclusion
from .inference import ModelUnavailable
from .longitudinal import (
    ComparisonRequest,
    CrossCalibration,
    LongitudinalMeasurement,
    LongitudinalRepository,
    LscProfile,
    baseline_candidates,
    compare_measurements,
)
from .preprocessing import ExcludedDicom, InvalidDicom, prepare_dicom
from .routing import InvalidProtocolOverride, registry_snapshot
from .service import UnsupportedProtocol, analyze
from .settings import settings
from third_party.dxa_pointplacement import UPSTREAM_COMMIT


app = FastAPI(
    title="Osseo AI DXA inference",
    version="0.1.0",
    description="Research inference service integrating hawaii-ai/dxa-pointplacement.",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=list(settings.allowed_origins),
    allow_credentials=False,
    allow_methods=["GET", "POST", "PUT"],
    allow_headers=["Content-Type", "X-Request-ID"],
    expose_headers=["X-Request-ID"],
)


REQUEST_ID = re.compile(r"^[A-Za-z0-9._:-]{8,128}$")


@app.middleware("http")
async def request_provenance(request: Request, call_next):
    supplied = request.headers.get("X-Request-ID", "")
    request_id = supplied if REQUEST_ID.fullmatch(supplied) else str(uuid.uuid4())
    request.state.request_id = request_id
    response = await call_next(request)
    response.headers["X-Request-ID"] = request_id
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["Referrer-Policy"] = "no-referrer"
    response.headers["Permissions-Policy"] = "camera=(), microphone=(), geolocation=()"
    if request.url.path.startswith("/api/"):
        response.headers["Cache-Control"] = "no-store"
    return response


def dataset_repository() -> DatasetRepository:
    return DatasetRepository(settings.dataset_root, settings.annotation_root)


def longitudinal_repository() -> LongitudinalRepository:
    return LongitudinalRepository(settings.longitudinal_root, dataset_repository())


@app.get("/health")
def health() -> dict[str, object]:
    try:
        dataset_summary = dataset_repository().summary()
        dataset_status: dict[str, object] = {"available": True, "studyCount": dataset_summary["studyCount"]}
    except (DatasetUnavailable, DatasetContractError, OSError, json.JSONDecodeError):
        dataset_status = {"available": False, "studyCount": 0}
    return {
        "status": "ok",
        "checkpointAvailable": settings.checkpoint_path.is_file(),
        "dataset": dataset_status,
        "upstreamCommit": UPSTREAM_COMMIT,
    }


@app.get("/api/v1/models")
def models() -> dict[str, object]:
    return {"models": registry_snapshot()}


@app.get("/api/v1/datasets/current")
def current_dataset() -> dict[str, object]:
    try:
        return dataset_repository().summary()
    except DatasetUnavailable as error:
        raise HTTPException(status_code=503, detail={"code": "DATASET_UNAVAILABLE", "message": str(error)}) from error
    except (DatasetContractError, OSError, json.JSONDecodeError) as error:
        raise HTTPException(status_code=500, detail={"code": "DATASET_INVALID", "message": str(error)}) from error


@app.get("/api/v1/datasets/current/integrity")
def current_dataset_integrity() -> dict[str, object]:
    try:
        return dataset_integrity(dataset_repository())
    except DatasetUnavailable as error:
        raise HTTPException(status_code=503, detail={"code": "DATASET_UNAVAILABLE", "message": str(error)}) from error
    except (DatasetContractError, OSError, ValueError) as error:
        raise HTTPException(status_code=500, detail={"code": "INTEGRITY_AUDIT_FAILED", "message": str(error)}) from error


@app.get("/api/v1/datasets/current/agreement")
def current_annotation_agreement() -> dict[str, object]:
    try:
        return annotation_agreement(dataset_repository())
    except DatasetUnavailable as error:
        raise HTTPException(status_code=503, detail={"code": "DATASET_UNAVAILABLE", "message": str(error)}) from error
    except (DatasetContractError, OSError, ValueError, json.JSONDecodeError) as error:
        raise HTTPException(status_code=500, detail={"code": "AGREEMENT_AUDIT_FAILED", "message": str(error)}) from error


@app.get("/api/v1/datasets/current/adjudication")
def current_adjudication_queue() -> dict[str, object]:
    try:
        report = annotation_agreement(dataset_repository())
        return {"studies": report["pendingAdjudication"], "count": report["pendingAdjudicationCount"]}
    except DatasetUnavailable as error:
        raise HTTPException(status_code=503, detail={"code": "DATASET_UNAVAILABLE", "message": str(error)}) from error
    except (DatasetContractError, OSError, ValueError, json.JSONDecodeError) as error:
        raise HTTPException(status_code=500, detail={"code": "AGREEMENT_AUDIT_FAILED", "message": str(error)}) from error


@app.get("/api/v1/readiness")
def readiness() -> dict[str, object]:
    try:
        return readiness_report(dataset_repository(), settings)
    except DatasetUnavailable as error:
        raise HTTPException(status_code=503, detail={"code": "DATASET_UNAVAILABLE", "message": str(error)}) from error
    except (DatasetContractError, OSError, ValueError, json.JSONDecodeError) as error:
        raise HTTPException(status_code=500, detail={"code": "READINESS_AUDIT_FAILED", "message": str(error)}) from error


@app.get("/api/v1/audit/status")
def audit_status() -> dict[str, object]:
    try:
        return verify_audit(settings.audit_log)
    except (OSError, ValueError, json.JSONDecodeError) as error:
        raise HTTPException(status_code=500, detail={"code": "AUDIT_TRAIL_INVALID", "message": str(error)}) from error


@app.get("/api/v1/longitudinal/lsc-profiles")
def longitudinal_lsc_profiles() -> dict[str, object]:
    try:
        values = [item.model_dump(mode="json") for item in longitudinal_repository().lsc_profiles()]
        return {"profiles": values, "count": len(values)}
    except (OSError, ValueError, json.JSONDecodeError) as error:
        raise HTTPException(status_code=500, detail={"code": "LONGITUDINAL_REGISTRY_INVALID", "message": str(error)}) from error


@app.put("/api/v1/longitudinal/lsc-profiles/{profile_id}")
def save_longitudinal_lsc_profile(profile_id: str, profile: LscProfile, request: Request) -> dict[str, object]:
    if profile.profileId != profile_id:
        raise HTTPException(status_code=400, detail={"code": "LSC_PROFILE_INVALID", "message": "Path and body profileId differ"})
    actor = pseudonymous_actor(profile.approvedBy, settings.audit_hmac_key)
    sanitized = profile.model_copy(update={"approvedBy": actor})
    try:
        payload = longitudinal_repository().save_lsc_profile(sanitized)
        append_audit(settings.audit_log, action="lsc_profile_saved", object_id=profile_id, request_id=request.state.request_id, actor_id=actor, details={"version": profile.version, "protocol": profile.protocol, "siteCount": len(profile.sites)})
        return payload
    except (OSError, ValueError, json.JSONDecodeError) as error:
        raise HTTPException(status_code=400, detail={"code": "LSC_PROFILE_INVALID", "message": str(error)}) from error


@app.get("/api/v1/longitudinal/cross-calibrations")
def longitudinal_cross_calibrations() -> dict[str, object]:
    try:
        values = [item.model_dump(mode="json") for item in longitudinal_repository().cross_calibrations()]
        return {"calibrations": values, "count": len(values)}
    except (OSError, ValueError, json.JSONDecodeError) as error:
        raise HTTPException(status_code=500, detail={"code": "LONGITUDINAL_REGISTRY_INVALID", "message": str(error)}) from error


@app.put("/api/v1/longitudinal/cross-calibrations/{calibration_id}")
def save_longitudinal_cross_calibration(calibration_id: str, calibration: CrossCalibration, request: Request) -> dict[str, object]:
    if calibration.calibrationId != calibration_id:
        raise HTTPException(status_code=400, detail={"code": "CROSS_CALIBRATION_INVALID", "message": "Path and body calibrationId differ"})
    actor = pseudonymous_actor(calibration.approvedBy, settings.audit_hmac_key)
    sanitized = calibration.model_copy(update={"approvedBy": actor})
    try:
        payload = longitudinal_repository().save_cross_calibration(sanitized)
        append_audit(settings.audit_log, action="cross_calibration_saved", object_id=calibration_id, request_id=request.state.request_id, actor_id=actor, details={"siteCount": len(calibration.sites)})
        return payload
    except (OSError, ValueError, json.JSONDecodeError) as error:
        raise HTTPException(status_code=400, detail={"code": "CROSS_CALIBRATION_INVALID", "message": str(error)}) from error


@app.put("/api/v1/longitudinal/measurements/{study_id}")
def save_longitudinal_measurement(study_id: str, measurement: LongitudinalMeasurement, request: Request) -> dict[str, object]:
    if measurement.studyId != study_id:
        raise HTTPException(status_code=400, detail={"code": "MEASUREMENT_INVALID", "message": "Path and body studyId differ"})
    actor = pseudonymous_actor(measurement.confirmedBy, settings.audit_hmac_key)
    sanitized = measurement.model_copy(update={"confirmedBy": actor})
    try:
        payload = longitudinal_repository().save_measurement(sanitized)
        append_audit(settings.audit_log, action="longitudinal_measurement_saved", object_id=study_id, request_id=request.state.request_id, actor_id=actor, details={"protocol": measurement.protocol, "source": measurement.source, "siteCount": len(measurement.sites)})
        return payload
    except KeyError as error:
        raise HTTPException(status_code=404, detail={"code": "STUDY_NOT_FOUND", "message": study_id}) from error
    except DatasetUnavailable as error:
        raise HTTPException(status_code=503, detail={"code": "DATASET_UNAVAILABLE", "message": str(error)}) from error
    except (DatasetContractError, OSError, ValueError, json.JSONDecodeError) as error:
        raise HTTPException(status_code=400, detail={"code": "MEASUREMENT_INVALID", "message": str(error)}) from error


@app.get("/api/v1/longitudinal/patients/{patient_group_id}/timeline")
def longitudinal_timeline(patient_group_id: str) -> dict[str, object]:
    if not re.fullmatch(r"PG-[A-F0-9]{20}", patient_group_id):
        raise HTTPException(status_code=400, detail={"code": "PATIENT_GROUP_INVALID", "message": patient_group_id})
    try:
        values = [item.model_dump(mode="json") for item in longitudinal_repository().measurements(patient_group_id)]
        return {"patientGroupId": patient_group_id, "measurements": values, "count": len(values)}
    except (OSError, ValueError, json.JSONDecodeError) as error:
        raise HTTPException(status_code=500, detail={"code": "LONGITUDINAL_REGISTRY_INVALID", "message": str(error)}) from error


@app.get("/api/v1/longitudinal/measurements/{study_id}/baseline-candidates")
def longitudinal_baseline_candidates(study_id: str) -> dict[str, object]:
    try:
        return baseline_candidates(longitudinal_repository(), study_id)
    except KeyError as error:
        raise HTTPException(status_code=404, detail={"code": "MEASUREMENT_NOT_FOUND", "message": study_id}) from error
    except (OSError, ValueError, json.JSONDecodeError) as error:
        raise HTTPException(status_code=400, detail={"code": "LONGITUDINAL_REGISTRY_INVALID", "message": str(error)}) from error


@app.post("/api/v1/longitudinal/compare")
def longitudinal_compare(comparison: ComparisonRequest, request: Request) -> dict[str, object]:
    try:
        result = compare_measurements(longitudinal_repository(), comparison)
        append_audit(settings.audit_log, action="longitudinal_comparison_computed", object_id=str(result["comparisonId"]), request_id=request.state.request_id, details={"baselineStudyId": comparison.baselineStudyId, "currentStudyId": comparison.currentStudyId, "status": result["status"], "lscProfileId": comparison.lscProfileId})
        return result
    except KeyError as error:
        raise HTTPException(status_code=404, detail={"code": "LONGITUDINAL_RECORD_NOT_FOUND", "message": str(error.args[0])}) from error
    except DatasetUnavailable as error:
        raise HTTPException(status_code=503, detail={"code": "DATASET_UNAVAILABLE", "message": str(error)}) from error
    except (DatasetContractError, OSError, ValueError, json.JSONDecodeError) as error:
        raise HTTPException(status_code=400, detail={"code": "COMPARISON_INVALID", "message": str(error)}) from error


@app.get("/api/v1/longitudinal/comparisons/{comparison_id}")
def longitudinal_comparison_record(comparison_id: str) -> dict[str, object]:
    if not re.fullmatch(r"CMP-[A-F0-9]{20}", comparison_id):
        raise HTTPException(status_code=400, detail={"code": "COMPARISON_ID_INVALID", "message": comparison_id})
    try:
        result = longitudinal_repository().comparison(comparison_id)
        if result is None:
            raise HTTPException(status_code=404, detail={"code": "COMPARISON_NOT_FOUND", "message": comparison_id})
        return result
    except HTTPException:
        raise
    except (OSError, ValueError, json.JSONDecodeError) as error:
        raise HTTPException(status_code=500, detail={"code": "LONGITUDINAL_REGISTRY_INVALID", "message": str(error)}) from error


@app.get("/api/v1/datasets/current/studies")
def dataset_studies(
    protocol: str | None = None,
    split: str | None = None,
    review_required: bool | None = Query(default=None),
    annotated: bool | None = Query(default=None),
    query: str = "",
) -> dict[str, object]:
    try:
        studies = dataset_repository().list_studies(
            protocol=protocol,
            split=split,
            review_required=review_required,
            annotated=annotated,
            query=query,
        )
        return {"studies": studies, "count": len(studies)}
    except DatasetUnavailable as error:
        raise HTTPException(status_code=503, detail={"code": "DATASET_UNAVAILABLE", "message": str(error)}) from error
    except (DatasetContractError, OSError, json.JSONDecodeError) as error:
        raise HTTPException(status_code=500, detail={"code": "DATASET_INVALID", "message": str(error)}) from error


@app.get("/api/v1/datasets/current/studies/{study_id}")
def dataset_study(study_id: str) -> dict[str, object]:
    try:
        return dataset_repository().get_study(study_id)
    except KeyError as error:
        raise HTTPException(status_code=404, detail={"code": "STUDY_NOT_FOUND", "message": study_id}) from error
    except DatasetUnavailable as error:
        raise HTTPException(status_code=503, detail={"code": "DATASET_UNAVAILABLE", "message": str(error)}) from error
    except (DatasetContractError, OSError, json.JSONDecodeError) as error:
        raise HTTPException(status_code=400, detail={"code": "DATASET_INVALID", "message": str(error)}) from error


@app.get("/api/v1/datasets/current/studies/{study_id}/assets/{asset_name}")
def dataset_asset(study_id: str, asset_name: str) -> FileResponse:
    try:
        return FileResponse(dataset_repository().asset_path(study_id, asset_name), media_type="image/png")
    except FileNotFoundError as error:
        raise HTTPException(status_code=404, detail={"code": "ASSET_NOT_FOUND", "message": asset_name}) from error
    except DatasetUnavailable as error:
        raise HTTPException(status_code=503, detail={"code": "DATASET_UNAVAILABLE", "message": str(error)}) from error
    except DatasetContractError as error:
        raise HTTPException(status_code=400, detail={"code": "DATASET_INVALID", "message": str(error)}) from error


@app.get("/api/v1/datasets/current/studies/{study_id}/raw/{channel}.png")
def dataset_raw_channel(study_id: str, channel: int) -> Response:
    try:
        return Response(dataset_repository().raw_channel_png(study_id, channel), media_type="image/png")
    except FileNotFoundError as error:
        raise HTTPException(status_code=404, detail={"code": "ASSET_NOT_FOUND", "message": "raw cube"}) from error
    except DatasetUnavailable as error:
        raise HTTPException(status_code=503, detail={"code": "DATASET_UNAVAILABLE", "message": str(error)}) from error
    except DatasetContractError as error:
        raise HTTPException(status_code=400, detail={"code": "DATASET_INVALID", "message": str(error)}) from error


@app.get("/api/v1/datasets/current/studies/{study_id}/annotations")
def dataset_annotations(study_id: str) -> dict[str, object]:
    try:
        values = dataset_repository().annotations(study_id)
        return {"annotations": values, "count": len(values)}
    except DatasetUnavailable as error:
        raise HTTPException(status_code=503, detail={"code": "DATASET_UNAVAILABLE", "message": str(error)}) from error
    except DatasetContractError as error:
        raise HTTPException(status_code=400, detail={"code": "ANNOTATION_INVALID", "message": str(error)}) from error


@app.put("/api/v1/datasets/current/studies/{study_id}/annotations")
def save_dataset_annotation(study_id: str, annotation: AnnotationDocument, request: Request) -> dict[str, object]:
    if annotation.studyId != study_id:
        raise HTTPException(status_code=400, detail={"code": "ANNOTATION_INVALID", "message": "Path and body studyId differ"})
    try:
        payload = dataset_repository().save_annotation(annotation)
        append_audit(
            settings.audit_log,
            action="annotation_saved",
            object_id=study_id,
            request_id=request.state.request_id,
            actor_id=pseudonymous_actor(annotation.expert.readerId, settings.audit_hmac_key),
            details={
                "readIndex": annotation.expert.readIndex,
                "adjudicated": annotation.expert.adjudicated,
                "overallAction": annotation.overallAction,
                "presentDefectCount": sum(item.present for item in annotation.defects),
            },
        )
        return payload
    except KeyError as error:
        raise HTTPException(status_code=404, detail={"code": "STUDY_NOT_FOUND", "message": study_id}) from error
    except DatasetUnavailable as error:
        raise HTTPException(status_code=503, detail={"code": "DATASET_UNAVAILABLE", "message": str(error)}) from error
    except DatasetContractError as error:
        raise HTTPException(status_code=400, detail={"code": "ANNOTATION_INVALID", "message": str(error)}) from error


@app.get("/api/v1/datasets/current/exports/coco")
def export_coco(request: Request) -> Response:
    try:
        payload = json.dumps(dataset_repository().coco_export(), ensure_ascii=False).encode("utf-8")
        append_audit(settings.audit_log, action="coco_exported", object_id="CURRENT_DATASET", request_id=request.state.request_id, details={"byteSize": len(payload)})
        return Response(payload, media_type="application/json", headers={"Content-Disposition": 'attachment; filename="osseo-annotations-coco.json"'})
    except DatasetUnavailable as error:
        raise HTTPException(status_code=503, detail={"code": "DATASET_UNAVAILABLE", "message": str(error)}) from error
    except DatasetContractError as error:
        raise HTTPException(status_code=400, detail={"code": "DATASET_INVALID", "message": str(error)}) from error


@app.get("/api/v1/datasets/current/exports/annotations")
def export_annotations(request: Request) -> Response:
    try:
        payload = dataset_repository().annotation_jsonl()
        append_audit(settings.audit_log, action="annotations_exported", object_id="CURRENT_DATASET", request_id=request.state.request_id, details={"byteSize": len(payload)})
        return Response(payload, media_type="application/x-ndjson", headers={"Content-Disposition": 'attachment; filename="osseo-annotations.jsonl"'})
    except DatasetUnavailable as error:
        raise HTTPException(status_code=503, detail={"code": "DATASET_UNAVAILABLE", "message": str(error)}) from error
    except DatasetContractError as error:
        raise HTTPException(status_code=400, detail={"code": "DATASET_INVALID", "message": str(error)}) from error


@app.get("/api/v1/exclusions")
def exclusions() -> dict[str, object]:
    try:
        values = read_exclusions(settings.exclusion_log)
        return {"exclusions": values, "count": len(values)}
    except (OSError, ValueError, json.JSONDecodeError) as error:
        raise HTTPException(status_code=500, detail={"code": "EXCLUSION_REGISTRY_INVALID", "message": str(error)}) from error


@app.post("/api/v1/studies/analyze")
async def analyze_study(
    request: Request,
    file: UploadFile = File(...),
    protocol_override: str | None = Form(default=None),
) -> dict[str, object]:
    filename = file.filename or "study.dcm"
    if not filename.lower().endswith((".dcm", ".dicom")) and file.content_type not in ("application/dicom", "application/dicom+json"):
        raise HTTPException(status_code=415, detail={"code": "UNSUPPORTED_MEDIA_TYPE", "message": "A DICOM file is required"})
    payload = await file.read(settings.max_upload_bytes + 1)
    if not payload:
        raise HTTPException(status_code=400, detail={"code": "EMPTY_FILE", "message": "The DICOM file is empty"})
    if len(payload) > settings.max_upload_bytes:
        raise HTTPException(status_code=413, detail={"code": "FILE_TOO_LARGE", "message": "The DICOM file exceeds the upload limit"})
    try:
        return analyze(prepare_dicom(payload), protocol_override)
    except ExcludedDicom as error:
        exclusion = record_exclusion(settings.exclusion_log, payload, error.reason)
        append_audit(settings.audit_log, action="presentation_object_excluded", object_id=str(exclusion["objectId"]), request_id=request.state.request_id, details={"reason": error.reason, "byteSize": len(payload)})
        raise HTTPException(
            status_code=422,
            detail={
                "code": "SECONDARY_CAPTURE_EXCLUDED",
                "message": str(error),
                "reason": error.reason,
                "trainingEligible": False,
                "exclusionId": exclusion["objectId"],
            },
        ) from error
    except InvalidDicom as error:
        raise HTTPException(status_code=400, detail={"code": "INVALID_DICOM", "message": str(error)}) from error
    except UnsupportedProtocol as error:
        raise HTTPException(status_code=422, detail={"code": "UNSUPPORTED_PROTOCOL", "message": str(error), "routing": error.decision.as_dict()}) from error
    except InvalidProtocolOverride as error:
        raise HTTPException(status_code=422, detail={"code": "INVALID_PROTOCOL_OVERRIDE", "message": str(error)}) from error
    except ModelUnavailable as error:
        raise HTTPException(status_code=503, detail={"code": "MODEL_UNAVAILABLE", "message": str(error)}) from error
