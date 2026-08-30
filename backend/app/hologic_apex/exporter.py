from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections import Counter
from pathlib import Path

from .loader import ApexDataset, ApexStudy, pseudonymize
from .parser import LOADER_VERSION, ProcessedImage
from .dataset_pipeline import dataset_qc_report, patient_grouped_splits, technical_qc


def _percentile(sorted_values: list[int], fraction: float) -> int:
    if not sorted_values:
        return 0
    return sorted_values[int((len(sorted_values) - 1) * fraction)]


def _processed_stats(image: ProcessedImage) -> dict[str, object]:
    values = sorted(image.pixels)
    clipped = sum(value in (0, 255) for value in image.pixels) / len(image.pixels)
    return {
        "tag": f"0x{image.tag:04X}",
        "width": image.width,
        "height": image.height,
        "descriptor": image.descriptor,
        "dtype": "uint8",
        "p01": _percentile(values, 0.01),
        "p99": _percentile(values, 0.99),
        "clippedFraction": round(clipped, 8),
    }


def _raw_stats(study: ApexStudy) -> dict[str, object]:
    raw = study.r_file
    phase_stats = []
    for index in range(raw.phase_count):
        values = sorted(raw.phase(index))
        phase_stats.append({
            "index": index,
            "minimum": values[0],
            "maximum": values[-1],
            "p01": _percentile(values, 0.01),
            "p99": _percentile(values, 0.99),
        })
    saturated = sum(value in (0, 65535) for value in raw.samples) / len(raw.samples)
    return {
        "storedWidth": raw.sample_width,
        "logicalWidth": raw.logical_width,
        "height": raw.height,
        "transmissionCount": raw.phase_count,
        "dtype": "uint16-le",
        "shape": [raw.height, raw.logical_width, raw.phase_count],
        "phaseSemantics": "unverified",
        "saturatedFraction": round(saturated, 8),
        "phases": phase_stats,
    }


def study_manifest(study: ApexStudy, key: bytes) -> dict[str, object]:
    study_id = "ST-" + pseudonymize(key, "study", study.patient_key + b"\0" + study.study_key)
    patient_id = "PG-" + pseudonymize(key, "patient", study.patient_key)
    device_id = "DV-" + pseudonymize(key, "device", study.device_key)
    return {
        "schemaVersion": "1.0.0",
        "loaderVersion": LOADER_VERSION,
        "studyId": study_id,
        "patientGroupId": patient_id,
        "deviceGroup": device_id,
        "protocol": study.protocol,
        "protocolCode": study.protocol_code,
        "acquisitionYear": study.acquisition_year,
        "softwareVersion": study.software_version,
        "source": {
            "pFingerprint": pseudonymize(key, "source-p", bytes.fromhex(study.p_file.sha256)),
            "rFingerprint": pseudonymize(key, "source-r", bytes.fromhex(study.r_file.sha256)),
        },
        "processedImages": [_processed_stats(image) for image in study.p_file.images],
        "raw": _raw_stats(study),
        "quality": {
            "reviewRequired": bool(study.flags),
            "flags": list(study.flags),
        },
        "privacy": {
            "pseudonymization": "HMAC-SHA256-160",
            "directIdentifiersExported": False,
        },
    }


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    path.chmod(0o600)


def _save_study_assets(target: Path, study: ApexStudy, manifest: dict[str, object]) -> None:
    import numpy as np
    from PIL import Image

    study_id = str(manifest["studyId"])
    image_dir = target / "processed" / study_id
    raw_dir = target / "raw" / study_id
    image_dir.mkdir(parents=True, mode=0o700)
    raw_dir.mkdir(parents=True, mode=0o700)
    for image in study.p_file.images:
        output = image_dir / f"p_{image.tag:04x}.png"
        Image.frombytes("L", (image.width, image.height), image.pixels).save(output, format="PNG", optimize=True)
        output.chmod(0o600)
    raw = np.asarray(study.r_file.samples, dtype=np.uint16).astype("<u2", copy=False).reshape(
        study.r_file.height,
        study.r_file.logical_width,
        study.r_file.phase_count,
    )
    output = raw_dir / "transmissions.npy"
    np.save(output, raw, allow_pickle=False)
    output.chmod(0o600)


def export_dataset(
    dataset: ApexDataset,
    output: Path,
    key: bytes,
    *,
    protocols: set[str] | None = None,
    split_seed: str = "osseo-dataset-v1",
) -> dict[str, object]:
    output = output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {output}")
    selected = tuple(study for study in dataset.studies if protocols is None or study.protocol in protocols)
    if not selected:
        raise ValueError("No studies match the selected protocol filter")
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".{output.name}-", dir=output.parent) as temporary:
        target = Path(temporary) / "dataset"
        target.mkdir(mode=0o700)
        manifests = [study_manifest(study, key) for study in selected]
        split_assignments = patient_grouped_splits(manifests, seed=split_seed)
        for manifest in manifests:
            manifest["split"] = split_assignments[str(manifest["patientGroupId"])]
            manifest["technicalQc"] = technical_qc(manifest)
        for study, manifest in zip(selected, manifests, strict=True):
            _save_study_assets(target, study, manifest)
        manifest_path = target / "manifest.jsonl"
        with manifest_path.open("w", encoding="utf-8") as stream:
            for manifest in manifests:
                stream.write(json.dumps(manifest, ensure_ascii=False, separators=(",", ":")) + "\n")
        manifest_path.chmod(0o600)
        split_counts = Counter(str(manifest["split"]) for manifest in manifests)
        _write_json(target / "splits.json", {
            "schemaVersion": "1.0.0",
            "strategy": "patient-grouped-deterministic",
            "seedFingerprint": hashlib.sha256(split_seed.encode("utf-8")).hexdigest()[:12],
            "counts": dict(sorted(split_counts.items())),
            "assignments": {str(manifest["studyId"]): str(manifest["split"]) for manifest in manifests},
        })
        qc_report = dataset_qc_report(manifests)
        _write_json(target / "qc_report.json", qc_report)
        summary = {
            **ApexDataset(selected).summary(key),
            "exportedStudyCount": len(selected),
            "reviewRequiredCount": sum(bool(study.flags) for study in selected),
            "assetCounts": {
                "processedPng": sum(len(study.p_file.images) for study in selected),
                "rawNpy": len(selected),
            },
            "splits": dict(sorted(split_counts.items())),
            "technicalQc": qc_report,
            "privacy": {
                "directIdentifiersExported": False,
                "sourceFilesCopied": False,
                "pseudonymization": "HMAC-SHA256-160",
            },
        }
        _write_json(target / "dataset_summary.json", summary)
        os.replace(target, output)
    return summary
