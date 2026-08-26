from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Literal


Protocol = Literal["spine", "hip", "total-body", "unsupported"]
ModelStatus = Literal["ready", "planned", "unsupported"]


class InvalidProtocolOverride(ValueError):
    """Raised only when a caller supplies an unknown manual protocol route."""


@dataclass(frozen=True)
class ModelDescriptor:
    key: str
    protocol: Protocol
    status: ModelStatus
    purpose: str


MODEL_REGISTRY: dict[Protocol, ModelDescriptor] = {
    "total-body": ModelDescriptor(
        key="hawaii-ai/dxa-pointplacement@7ac19eb",
        protocol="total-body",
        status="ready",
        purpose="105 total-body anatomical landmarks and geometry features",
    ),
    "spine": ModelDescriptor(
        key="osseo/spine-multitask@planned",
        protocol="spine",
        status="planned",
        purpose="L1-L4 segmentation, ROI validation, positioning and defect classification",
    ),
    "hip": ModelDescriptor(
        key="osseo/hip-multitask@planned",
        protocol="hip",
        status="planned",
        purpose="Proximal femur segmentation, ROI validation, rotation and defect classification",
    ),
    "unsupported": ModelDescriptor(
        key="none",
        protocol="unsupported",
        status="unsupported",
        purpose="No anatomical model is selected",
    ),
}


@dataclass(frozen=True)
class RouteDecision:
    protocol: Protocol
    confidence: float
    source: Literal["dicom-rules", "manual-override", "ambiguous"]
    evidence: tuple[str, ...]
    model: ModelDescriptor

    def as_dict(self) -> dict[str, object]:
        return {
            "protocol": self.protocol,
            "confidence": round(self.confidence, 4),
            "source": self.source,
            "evidence": list(self.evidence),
            "modelKey": self.model.key,
            "modelStatus": self.model.status,
        }


PATTERNS: dict[Protocol, tuple[re.Pattern[str], ...]] = {
    "total-body": tuple(re.compile(pattern, re.I) for pattern in (
        r"\bTOTAL[ _-]?BODY\b", r"\bWHOLE[ _-]?BODY\b", r"ВС[ЕЁ]\s+ТЕЛО", r"ВСЕГО\s+ТЕЛА",
    )),
    "spine": tuple(re.compile(pattern, re.I) for pattern in (
        r"\bLSPINE\b", r"\bLUMBAR\b", r"\bSPINE\b", r"\bL[1-5](?:\s*[-–]\s*L?[1-5])?\b", r"ПОЯСНИЧ", r"ПОЗВОН",
    )),
    "hip": tuple(re.compile(pattern, re.I) for pattern in (
        r"\bHIP\b", r"\bFEMUR\b", r"\bFEMOR", r"PROXIMAL\s+FEM", r"БЕДР", r"ТАЗОБЕДР",
    )),
}

FIELD_WEIGHTS = {
    "body_part": 0.55,
    "protocol_name": 0.30,
    "description": 0.15,
}


def _override(value: str) -> Protocol:
    normalized = value.strip().lower().replace("_", "-")
    aliases: dict[str, Protocol] = {
        "spine": "spine", "spine-pa": "spine", "lumbar": "spine",
        "hip": "hip", "hip-left": "hip", "hip-right": "hip",
        "total-body": "total-body", "whole-body": "total-body", "totalbody": "total-body",
        "unsupported": "unsupported",
    }
    if normalized not in aliases:
        raise InvalidProtocolOverride(f"Unknown protocol override: {value}")
    return aliases[normalized]


def route_protocol(metadata: dict[str, Any], protocol_override: str | None = None) -> RouteDecision:
    if protocol_override:
        protocol = _override(protocol_override)
        return RouteDecision(
            protocol=protocol,
            confidence=1.0,
            source="manual-override",
            evidence=(f"manual:{protocol_override}",),
            model=MODEL_REGISTRY[protocol],
        )

    scores: dict[Protocol, float] = {"spine": 0.0, "hip": 0.0, "total-body": 0.0, "unsupported": 0.0}
    evidence: dict[Protocol, list[str]] = {key: [] for key in scores}
    for field, weight in FIELD_WEIGHTS.items():
        value = str(metadata.get(field, "") or "").strip()
        if not value:
            continue
        for protocol, patterns in PATTERNS.items():
            match = next((pattern.search(value) for pattern in patterns if pattern.search(value)), None)
            if match:
                scores[protocol] += weight
                evidence[protocol].append(f"{field}:{match.group(0)}")

    ranked = sorted((score, protocol) for protocol, score in scores.items() if protocol != "unsupported")
    best_score, best_protocol = ranked[-1]
    second_score = ranked[-2][0]
    margin = best_score - second_score
    if best_score < 0.15 or margin < 0.10:
        combined = tuple(item for values in evidence.values() for item in values)
        return RouteDecision("unsupported", max(0.0, min(1.0, margin)), "ambiguous", combined, MODEL_REGISTRY["unsupported"])

    confidence = min(1.0, best_score * 0.75 + margin * 0.25)
    return RouteDecision(best_protocol, confidence, "dicom-rules", tuple(evidence[best_protocol]), MODEL_REGISTRY[best_protocol])


def registry_snapshot() -> list[dict[str, str]]:
    return [
        {
            "key": descriptor.key,
            "protocol": descriptor.protocol,
            "status": descriptor.status,
            "purpose": descriptor.purpose,
        }
        for descriptor in MODEL_REGISTRY.values()
        if descriptor.protocol != "unsupported"
    ]
