from __future__ import annotations

import hashlib
import itertools
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from PIL import Image

from .audit import verify_audit
from .dataset import DatasetRepository
from .routing import registry_snapshot
from .settings import Settings


RELEASE_POLICY_VERSION = "ru-dxa-qc/1.0.0"
CLINICAL_EVIDENCE_ID = re.compile(r"^CVR-[A-Z0-9._-]{8,64}$")
ACCEPTED_AUTH_MODES = {"oidc", "trusted-proxy"}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _difference_hash(path: Path) -> int:
    with Image.open(path) as image:
        pixels = list(image.convert("L").resize((9, 8), Image.Resampling.BILINEAR).getdata())
    value = 0
    for row in range(8):
        for column in range(8):
            value = (value << 1) | int(pixels[row * 9 + column] > pixels[row * 9 + column + 1])
    return value


def dataset_integrity(repository: DatasetRepository) -> dict[str, Any]:
    records = repository.records()
    root = repository.root
    if root is None:
        raise RuntimeError("Dataset root is unavailable")
    manifest_hash = _sha256(root / "manifest.jsonl")
    split_by_patient: dict[str, set[str]] = defaultdict(set)
    exact_assets: dict[str, list[dict[str, str]]] = defaultdict(list)
    source_fingerprints: dict[str, list[dict[str, str]]] = defaultdict(list)
    perceptual_assets: list[dict[str, Any]] = []
    missing_assets: list[str] = []
    malformed_assets: list[str] = []

    for record in records:
        study_id = str(record["studyId"])
        split = str(record.get("split", "unassigned"))
        patient_group_id = str(record["patientGroupId"])
        split_by_patient[patient_group_id].add(split)
        for fingerprint in record.get("source", {}).values():
            source_fingerprints[str(fingerprint)].append({"studyId": study_id, "split": split})
        for image in record.get("processedImages", []):
            tag = str(image["tag"]).removeprefix("0x").lower()
            path = root / "processed" / study_id / f"p_{tag}.png"
            if not path.is_file():
                missing_assets.append(str(path.relative_to(root)))
                continue
            try:
                with Image.open(path) as asset:
                    width, height = asset.size
                if width != int(image["width"]) or height != int(image["height"]):
                    malformed_assets.append(f"{study_id}:p_{tag}.png:dimension_mismatch")
                checksum = _sha256(path)
                item = {"studyId": study_id, "patientGroupId": patient_group_id, "split": split, "asset": f"p_{tag}.png"}
                exact_assets[checksum].append(item)
                perceptual_assets.append({**item, "hash": _difference_hash(path)})
            except (OSError, ValueError):
                malformed_assets.append(f"{study_id}:p_{tag}.png:unreadable")
        raw_path = root / "raw" / study_id / "transmissions.npy"
        if not raw_path.is_file():
            missing_assets.append(str(raw_path.relative_to(root)))
        else:
            try:
                import numpy as np

                cube = np.load(raw_path, mmap_mode="r", allow_pickle=False)
                expected_shape = tuple(int(value) for value in record.get("raw", {}).get("shape", []))
                if cube.dtype != np.uint16 or cube.ndim != 3 or tuple(cube.shape) != expected_shape or cube.shape[2] != 6:
                    malformed_assets.append(f"{study_id}:transmissions.npy:contract_mismatch")
            except (OSError, ValueError):
                malformed_assets.append(f"{study_id}:transmissions.npy:unreadable")

    patient_leakage = [
        {"patientGroupId": patient, "splits": sorted(splits)}
        for patient, splits in split_by_patient.items() if len(splits) > 1
    ]
    exact_cross_split = [
        {"sha256": checksum, "occurrences": values}
        for checksum, values in exact_assets.items()
        if len({value["split"] for value in values}) > 1
    ]
    source_cross_split = [
        {"fingerprint": fingerprint, "occurrences": values}
        for fingerprint, values in source_fingerprints.items()
        if len({value["split"] for value in values}) > 1
    ]
    near_cross_split: list[dict[str, Any]] = []
    for first, second in itertools.combinations(perceptual_assets, 2):
        if first["split"] == second["split"] or first["patientGroupId"] == second["patientGroupId"]:
            continue
        distance = int(first["hash"] ^ second["hash"]).bit_count()
        if distance <= 3:
            near_cross_split.append({
                "firstStudyId": first["studyId"],
                "secondStudyId": second["studyId"],
                "hammingDistance": distance,
                "splits": [first["split"], second["split"]],
            })
    blockers = len(missing_assets) + len(malformed_assets) + len(patient_leakage) + len(exact_cross_split) + len(source_cross_split)
    return {
        "schemaVersion": "1.0.0",
        "datasetVersion": f"DS-{manifest_hash[:16].upper()}",
        "manifestSha256": manifest_hash,
        "status": "pass" if blockers == 0 else "fail",
        "studyCount": len(records),
        "patientGroupCount": len(split_by_patient),
        "processedAssetCount": len(perceptual_assets),
        "checks": {
            "missingAssets": missing_assets,
            "malformedAssets": malformed_assets,
            "patientLeakage": patient_leakage,
            "exactDuplicateCrossSplit": exact_cross_split,
            "sourceFingerprintCrossSplit": source_cross_split,
            "nearDuplicateCrossSplit": near_cross_split,
        },
        "blockerCount": blockers,
        "warningCount": len(near_cross_split),
    }


def _latest_by_reader(documents: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    latest_by_reader: dict[str, dict[str, Any]] = {}
    for document in documents:
        reader = str(document.get("expert", {}).get("readerId", ""))
        if reader and reader not in latest_by_reader:
            latest_by_reader[reader] = document
    return latest_by_reader


def _pairwise_documents(documents: list[dict[str, Any]]) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    independent_reads = [
        document for document in _latest_by_reader(documents).values()
        if not bool(document.get("expert", {}).get("adjudicated"))
    ]
    return list(itertools.combinations(independent_reads, 2))


def _cohen_kappa(pairs: list[tuple[str, str]]) -> float | None:
    if not pairs:
        return None
    labels = sorted({value for pair in pairs for value in pair})
    observed = sum(first == second for first, second in pairs) / len(pairs)
    first_counts = Counter(first for first, _ in pairs)
    second_counts = Counter(second for _, second in pairs)
    expected = sum(first_counts[label] * second_counts[label] for label in labels) / (len(pairs) ** 2)
    if math.isclose(expected, 1.0):
        return 1.0 if math.isclose(observed, 1.0) else None
    return round((observed - expected) / (1 - expected), 6)


def annotation_agreement(repository: DatasetRepository) -> dict[str, Any]:
    records = repository.records()
    action_pairs: list[tuple[str, str]] = []
    defect_pairs: dict[str, list[tuple[str, str]]] = defaultdict(list)
    landmark_errors: list[float] = []
    pending: list[dict[str, Any]] = []
    annotated = 0
    double_read = 0
    adjudicated = 0
    total_reads = 0
    for record in records:
        study_id = str(record["studyId"])
        documents = repository.annotations(study_id)
        latest_by_reader = _latest_by_reader(documents)
        unique_reader_count = len(latest_by_reader)
        has_valid_adjudication = unique_reader_count >= 3 and any(
            bool(item.get("expert", {}).get("adjudicated"))
            for item in latest_by_reader.values()
        )
        total_reads += len(documents)
        annotated += int(bool(documents))
        adjudicated += int(has_valid_adjudication)
        pairs = _pairwise_documents(documents)
        if not pairs:
            continue
        double_read += 1
        study_disagreement = False
        for first, second in pairs:
            first_action, second_action = str(first.get("overallAction")), str(second.get("overallAction"))
            action_pairs.append((first_action, second_action))
            study_disagreement |= first_action != second_action
            first_defects = {str(item["code"]): item for item in first.get("defects", [])}
            second_defects = {str(item["code"]): item for item in second.get("defects", [])}
            for code in sorted(set(first_defects) | set(second_defects)):
                first_label = f"{bool(first_defects.get(code, {}).get('present'))}:{first_defects.get(code, {}).get('severity', 'none')}"
                second_label = f"{bool(second_defects.get(code, {}).get('present'))}:{second_defects.get(code, {}).get('severity', 'none')}"
                defect_pairs[code].append((first_label, second_label))
                study_disagreement |= first_label != second_label
            first_landmarks = {str(item["name"]): item for item in first.get("landmarks", []) if item.get("visible")}
            second_landmarks = {str(item["name"]): item for item in second.get("landmarks", []) if item.get("visible")}
            for name in set(first_landmarks) & set(second_landmarks):
                dx = float(first_landmarks[name]["x"]) - float(second_landmarks[name]["x"])
                dy = float(first_landmarks[name]["y"]) - float(second_landmarks[name]["y"])
                landmark_errors.append(math.hypot(dx, dy))
        if study_disagreement and not has_valid_adjudication:
            pending.append({"studyId": study_id, "protocol": record.get("protocol"), "readerCount": unique_reader_count, "reason": "reader_disagreement"})
    action_agreement = sum(first == second for first, second in action_pairs) / len(action_pairs) if action_pairs else None
    return {
        "schemaVersion": "1.0.0",
        "studyCount": len(records),
        "annotatedStudyCount": annotated,
        "annotationCoverage": round(annotated / len(records), 6) if records else 0,
        "expertReadCount": total_reads,
        "doubleReadStudyCount": double_read,
        "doubleReadCoverage": round(double_read / len(records), 6) if records else 0,
        "adjudicatedStudyCount": adjudicated,
        "pendingAdjudicationCount": len(pending),
        "overallAction": {
            "comparisonCount": len(action_pairs),
            "percentAgreement": round(action_agreement, 6) if action_agreement is not None else None,
            "cohenKappa": _cohen_kappa(action_pairs),
        },
        "defects": {
            code: {
                "comparisonCount": len(pairs),
                "percentAgreement": round(sum(first == second for first, second in pairs) / len(pairs), 6),
                "cohenKappa": _cohen_kappa(pairs),
            }
            for code, pairs in sorted(defect_pairs.items())
        },
        "landmarks": {
            "comparisonCount": len(landmark_errors),
            "meanNormalizedDistance": round(sum(landmark_errors) / len(landmark_errors), 6) if landmark_errors else None,
        },
        "pendingAdjudication": pending,
    }


def _gate(identifier: str, label: str, status: str, evidence: str, next_action: str, *, critical: bool = True) -> dict[str, Any]:
    return {"id": identifier, "label": label, "status": status, "critical": critical, "evidence": evidence, "nextAction": next_action}


def readiness_report(repository: DatasetRepository, settings: Settings) -> dict[str, Any]:
    integrity = dataset_integrity(repository)
    agreement = annotation_agreement(repository)
    records = repository.records()
    flagged = [item for item in records if item.get("technicalQc", {}).get("requiresExpertReview") or item.get("quality", {}).get("reviewRequired")]
    reviewed_flagged = sum(bool(repository.annotations(str(item["studyId"]))) for item in flagged)
    registry = {str(item["protocol"]): str(item["status"]) for item in registry_snapshot()}
    audit = verify_audit(settings.audit_log)
    gates = [
        _gate("dataset_integrity", "Целостность датасета", "pass" if integrity["status"] == "pass" else "block", f"blockers={integrity['blockerCount']}; warnings={integrity['warningCount']}", "Устранить отсутствующие ассеты, нарушения split и дубликаты."),
        _gate("patient_leakage", "Patient-level leakage", "pass" if not integrity["checks"]["patientLeakage"] else "block", f"cross-split patients={len(integrity['checks']['patientLeakage'])}", "Пересобрать split только по patientGroupId."),
        _gate("annotation_coverage", "Экспертная разметка", "pass" if agreement["annotationCoverage"] >= 0.8 else "block", f"coverage={agreement['annotationCoverage']:.1%}", "Разметить не менее 80% pilot-выборки по labelbook."),
        _gate("double_read", "Двойное чтение", "pass" if agreement["doubleReadCoverage"] >= 0.2 else "block", f"coverage={agreement['doubleReadCoverage']:.1%}", "Назначить независимое второе чтение минимум 20% исследований."),
        _gate("adjudication", "Разрешение расхождений", "pass" if agreement["doubleReadStudyCount"] > 0 and agreement["pendingAdjudicationCount"] == 0 else "block", f"pending={agreement['pendingAdjudicationCount']}; double-read={agreement['doubleReadStudyCount']}", "Провести adjudication третьим экспертом для всех расхождений."),
        _gate("technical_review", "Ручной разбор технических флагов", "pass" if len(flagged) == reviewed_flagged else "block", f"reviewed={reviewed_flagged}/{len(flagged)}", "Разметить все исследования, отмеченные technical baseline."),
        _gate("spine_model", "Профильная модель PA spine", "pass" if registry.get("spine") == "ready" else "block", f"registry={registry.get('spine', 'missing')}", "Обучить, откалибровать и заморозить spine-модель на patient-level holdout."),
        _gate("hip_model", "Профильная модель hip", "pass" if registry.get("hip") == "ready" else "block", f"registry={registry.get('hip', 'missing')}", "Обучить и независимо валидировать hip-модель."),
        _gate("audit_chain", "Контроль целостности audit trail", "pass" if audit["valid"] and audit["eventCount"] > 0 else "block", f"valid={audit['valid']}; events={audit['eventCount']}", "Выполнить и проверить pilot-сценарии сохранения и экспорта."),
        _gate("audit_key", "Ключ псевдонимизации audit", "pass" if len(settings.audit_hmac_key.encode("utf-8")) >= 16 else "block", f"configured={bool(settings.audit_hmac_key)}", "Настроить отдельный OSSEO_AUDIT_HMAC_KEY длиной не менее 16 байт через secret storage."),
        _gate("access_control", "Контроль доступа", "pass" if settings.auth_mode in ACCEPTED_AUTH_MODES else "block", f"mode={settings.auth_mode}", "Подключить OIDC/reverse-proxy RBAC перед работой с клиническими данными."),
        _gate("clinical_validation", "Независимая клиническая валидация", "pass" if CLINICAL_EVIDENCE_ID.fullmatch(settings.clinical_validation_id) else "block", f"evidence={settings.clinical_validation_id or 'absent'}", "Провести silent-mode исследование и зафиксировать идентификатор отчёта формата CVR-…"),
        _gate("raw_semantics", "Физика шести raw-каналов", "warn", "phaseSemantics=unverified", "Подтвердить порядок каналов на phantom/vendor reference до расчёта BMD.", critical=False),
    ]
    critical_blockers = sum(item["critical"] and item["status"] == "block" for item in gates)
    return {
        "schemaVersion": "1.0.0",
        "policyVersion": RELEASE_POLICY_VERSION,
        "intendedUse": {
            "ru": "Исследовательская система поддержки контроля качества получения и анализа DXA; не ставит диагноз и не принимает автономных клинических решений.",
            "en": "Research decision support for DXA acquisition and analysis quality control; no diagnosis or autonomous clinical decision.",
        },
        "datasetVersion": integrity["datasetVersion"],
        "stage": "silent-pilot-ready" if critical_blockers == 0 else "research",
        "criticalBlockerCount": critical_blockers,
        "warningCount": sum(item["status"] == "warn" for item in gates),
        "gates": gates,
        "integrity": integrity,
        "agreement": agreement,
        "audit": audit,
    }
