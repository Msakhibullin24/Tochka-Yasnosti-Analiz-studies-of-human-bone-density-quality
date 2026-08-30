from __future__ import annotations

import hashlib
from collections import Counter
from typing import Any


def patient_grouped_splits(manifests: list[dict[str, Any]], *, seed: str) -> dict[str, str]:
    groups = sorted({str(item["patientGroupId"]) for item in manifests})
    ranked = sorted(groups, key=lambda group: hashlib.sha256(f"{seed}\0{group}".encode()).digest())
    count = len(ranked)
    if count == 0:
        return {}
    validation_count = 1 if count >= 2 else 0
    test_count = max(1, round(count * 0.15)) if count >= 5 else 0
    train_count = max(1, count - validation_count - test_count)
    assignments: dict[str, str] = {}
    for index, group in enumerate(ranked):
        assignments[group] = "train" if index < train_count else "validation" if index < train_count + validation_count else "test"
    return assignments


def technical_qc(manifest: dict[str, Any]) -> dict[str, Any]:
    flags = list(manifest.get("quality", {}).get("flags", []))
    image_metrics: list[dict[str, Any]] = []
    for image in manifest.get("processedImages", []):
        dynamic_range = max(0.0, min(1.0, (float(image["p99"]) - float(image["p01"])) / 255.0))
        clipped = float(image["clippedFraction"])
        if dynamic_range < 0.15:
            flags.append(f"processed_image_{image['tag']}_low_dynamic_range")
        if clipped > 0.02:
            flags.append(f"processed_image_{image['tag']}_clipped")
        image_metrics.append({
            "tag": image["tag"],
            "dynamicRangeFraction": round(dynamic_range, 6),
            "clippedFraction": clipped,
            "aspectRatio": round(float(image["width"]) / max(1.0, float(image["height"])), 6),
        })
    raw_saturation = float(manifest.get("raw", {}).get("saturatedFraction", 0.0))
    if raw_saturation > 0.001:
        flags.append("raw_signal_saturated")
    flags = sorted(set(flags))
    penalty = min(80, sum(20 if "saturated" in flag or "short" in flag else 10 for flag in flags))
    supported = manifest.get("protocol") in {"spine_pa", "hip_left", "hip_right", "total_body"}
    return {
        "baselineVersion": "osseo-technical-qc/1.0.0",
        "score": max(0, 100 - penalty),
        "trainingCandidate": supported and not any("saturated" in flag for flag in flags),
        "requiresExpertReview": bool(flags),
        "flags": flags,
        "processed": image_metrics,
        "raw": {"saturatedFraction": raw_saturation, "phaseSemantics": manifest.get("raw", {}).get("phaseSemantics", "unverified")},
        "limitations": [
            "Technical baseline only; no anatomical or clinical conclusion.",
            "Raw channel physics remain unverified.",
        ],
    }


def dataset_qc_report(manifests: list[dict[str, Any]]) -> dict[str, Any]:
    flags = Counter(flag for item in manifests for flag in item.get("technicalQc", {}).get("flags", []))
    return {
        "schemaVersion": "1.0.0",
        "baselineVersion": "osseo-technical-qc/1.0.0",
        "studyCount": len(manifests),
        "trainingCandidateCount": sum(bool(item.get("technicalQc", {}).get("trainingCandidate")) for item in manifests),
        "expertReviewCount": sum(bool(item.get("technicalQc", {}).get("requiresExpertReview")) for item in manifests),
        "flags": dict(sorted(flags.items())),
        "phaseSemantics": "unverified",
    }
