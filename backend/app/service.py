from __future__ import annotations

import hashlib
from datetime import datetime, timezone

from .inference import model
from .preprocessing import PreparedStudy
from .qc import assess_landmarks, summarize, text
from .routing import RouteDecision, route_protocol
from .settings import settings


class UnsupportedProtocol(ValueError):
    def __init__(self, message: str, decision: RouteDecision):
        super().__init__(message)
        self.decision = decision


def _hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:8].upper()


def _format_date(date: str, time: str) -> str:
    date_label = f"{date[6:8]}.{date[4:6]}.{date[:4]}" if len(date) == 8 and date.isdigit() else "—"
    time_label = f"{time[:2]}:{time[2:4]}" if len(time) >= 4 and time[:4].isdigit() else "—"
    return f"{date_label} · {time_label}"


def analyze(prepared: PreparedStudy, protocol_override: str | None = None) -> dict[str, object]:
    metadata = prepared.metadata
    decision = route_protocol(metadata, protocol_override)
    if decision.protocol != "total-body" or decision.model.status != "ready":
        raise UnsupportedProtocol(
            f"No ready anatomical model for protocol '{decision.protocol}'.",
            decision,
        )
    landmarks = model.predict(prepared.image)
    criteria = assess_landmarks(landmarks, prepared.content_bbox)
    status, score, confidence = summarize(criteria)

    identity_seed = metadata.get("series_uid") or metadata.get("study_uid") or hashlib.sha256(prepared.image.tobytes()).hexdigest()
    patient_seed = metadata.get("patient_id") or metadata.get("study_uid") or identity_seed
    identifier = _hash(str(identity_seed))
    patient_identifier = _hash(str(patient_seed))
    accession_identifier = _hash(str(metadata.get("accession_number") or identity_seed))
    series_identifier = _hash(str(metadata.get("series_uid") or identity_seed))
    study_identifier = _hash(str(metadata.get("study_uid") or identity_seed))
    privacy_verified = bool(metadata.get("patient_identity_removed"))

    warnings = [
        text(
            "Исследовательская модель обучена на извлечённых air-ratio total-body DXA; текущий DICOM нормализован по процентилям и требует локальной валидации.",
            "The research model was trained on extracted air-ratio total-body DXA; this DICOM was percentile-normalized and requires local validation.",
        ),
        text(
            "Модель создаёт 105 ориентиров, но не проверяет vendor-specific ROI и не диагностирует остеопороз.",
            "The model creates 105 landmarks but does not validate vendor-specific ROIs or diagnose osteoporosis.",
        ),
    ]
    if not privacy_verified:
        warnings.append(text(
            "DICOM не подтверждает удаление идентификаторов пациента; ответ содержит только псевдонимы.",
            "The DICOM does not confirm identity removal; the response contains pseudonyms only.",
        ))

    recommendation = text(
        "Выявлены нарушения укладки; подтвердите результат и необходимость повторного исследования экспертно."
        if status == "rejected" else
        "Проверьте пограничные критерии и артефакты перед подтверждением исследования.",
        "Positioning issues were detected; confirm the result and need for repeat acquisition by expert review."
        if status == "rejected" else
        "Review borderline criteria and artifacts before approving the study.",
    )
    return {
        "id": f"ST-{identifier}",
        "patientId": f"P-{patient_identifier}",
        "filename": f"DICOM-{identifier}.dcm",
        "acquiredAt": _format_date(str(metadata.get("study_date", "")), str(metadata.get("study_time", ""))) if privacy_verified else "—",
        "type": "total-body",
        "status": status,
        "score": score,
        "confidence": confidence,
        "device": " ".join(filter(None, (metadata.get("manufacturer"), metadata.get("model_name")))) or "Не указано",
        "institution": text("Скрыто" if metadata.get("institution_present") else "Не указано", "Masked" if metadata.get("institution_present") else "Not provided"),
        "operator": "MASKED" if metadata.get("operator_present") else "—",
        "accessionNumber": f"ACC-{accession_identifier}",
        "seriesUid": f"UID-{series_identifier}",
        "studyInstanceUid": f"UID-{study_identifier}" if metadata.get("study_uid") else None,
        "technical": {
            "modality": metadata.get("modality") or "OT",
            "rows": prepared.height,
            "columns": prepared.width,
            "pixelSpacing": f"{metadata['pixel_spacing']} mm" if metadata.get("pixel_spacing") else None,
            "photometricInterpretation": metadata.get("photometric") or None,
            "transferSyntaxUid": metadata.get("transfer_syntax_uid") or None,
            "bitsAllocated": metadata.get("bits_allocated") or None,
        },
        "provenance": {
            "mode": "research-model",
            "modelVersion": settings.model_version,
            "criteriaVersion": settings.criteria_version,
            "processedAt": datetime.now(timezone.utc).isoformat(),
            "warnings": warnings,
        },
        "privacy": {
            "deidentificationVerified": privacy_verified,
            "burnedInAnnotation": metadata.get("burned_in_annotation") if metadata.get("burned_in_annotation") in ("YES", "NO") else None,
        },
        "previewUrl": prepared.preview_url,
        "routing": decision.as_dict(),
        "landmarks": [item.as_dict() for item in landmarks],
        "criteria": [item.as_dict() for item in criteria],
        "recommendation": recommendation,
    }
