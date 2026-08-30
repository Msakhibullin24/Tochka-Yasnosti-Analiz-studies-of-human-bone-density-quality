from __future__ import annotations

import hashlib
import hmac
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from .parser import (
    ApexFormatError,
    ApexPFile,
    ApexRFile,
    LOADER_VERSION,
    ProcessedImage,
    decode_text,
    parse_p_file,
    parse_r_file,
)


PAIR_PATTERN = re.compile(r"^(?P<stem>.+)\.(?P<kind>[pr])(?P<suffix>[a-z0-9]{2})$", re.IGNORECASE)
YEAR_PATTERN = re.compile(r"(?:19|20)\d{2}")


@dataclass(frozen=True)
class ApexPair:
    key: str
    p_path: Path = field(repr=False)
    r_path: Path = field(repr=False)


@dataclass(frozen=True)
class ApexStudy:
    pair_key: str = field(repr=False)
    p_file: ApexPFile = field(repr=False)
    r_file: ApexRFile = field(repr=False)
    protocol: str
    protocol_code: str
    acquisition_year: str | None
    software_version: str | None
    patient_key: bytes = field(repr=False)
    study_key: bytes = field(repr=False)
    device_key: bytes = field(repr=False)
    flags: tuple[str, ...]


@dataclass(frozen=True)
class ApexDataset:
    studies: tuple[ApexStudy, ...]

    def summary(self, pseudonym_key: bytes) -> dict[str, object]:
        protocols = Counter(study.protocol for study in self.studies)
        years = Counter(study.acquisition_year or "unknown" for study in self.studies)
        versions = Counter(study.software_version or "unknown" for study in self.studies)
        flags = Counter(flag for study in self.studies for flag in study.flags)
        patient_groups = {
            pseudonymize(pseudonym_key, "patient", study.patient_key)
            for study in self.studies
        }
        return {
            "schemaVersion": "1.0.0",
            "loaderVersion": LOADER_VERSION,
            "studyCount": len(self.studies),
            "patientGroupCount": len(patient_groups),
            "protocols": dict(sorted(protocols.items())),
            "acquisitionYears": dict(sorted(years.items())),
            "softwareVersions": dict(sorted(versions.items())),
            "qualityFlags": dict(sorted(flags.items())),
            "rawLayout": {
                "dtype": "uint16-le",
                "shape": "height × logical_width × 6",
                "phaseSemantics": "unverified",
            },
        }


def pseudonymize(key: bytes, namespace: str, value: bytes) -> str:
    if len(key) < 16:
        raise ValueError("Pseudonymization key must contain at least 16 bytes")
    digest = hmac.new(key, namespace.encode("ascii") + b"\0" + value, hashlib.sha256).hexdigest()
    return digest[:20].upper()


def discover_pairs(root: Path) -> tuple[ApexPair, ...]:
    grouped: dict[str, dict[str, Path]] = {}
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        if path.suffix.lower() == ".rar":
            continue
        match = PAIR_PATTERN.match(path.name)
        if not match:
            continue
        relative_parent = path.parent.relative_to(root).as_posix()
        identity = f"{relative_parent}/{match.group('stem').lower()}.{match.group('suffix').lower()}"
        kind = match.group("kind").lower()
        if kind in grouped.setdefault(identity, {}):
            raise ApexFormatError(f"Duplicate {kind.upper()} file for pair {identity}")
        grouped[identity][kind] = path
    if not grouped:
        raise ApexFormatError(f"No Hologic P/R files found under {root}")
    missing = [key for key, value in grouped.items() if set(value) != {"p", "r"}]
    if missing:
        preview = ", ".join(missing[:5])
        raise ApexFormatError(f"Unpaired Hologic scan file(s): {preview}")
    return tuple(ApexPair(key, value["p"], value["r"]) for key, value in sorted(grouped.items()))


def _protocol(code: str) -> str:
    normalized = Path(code.replace("\\", "/")).name.upper()
    if re.fullmatch(r"A[PS]\d+SP[A-Z0-9]*\.PRO", normalized):
        return "spine_pa"
    if re.fullmatch(r"A[PS]\d+LH[A-Z0-9]*\.PRO", normalized):
        return "hip_left"
    if re.fullmatch(r"A[PS]\d+RH[A-Z0-9]*\.PRO", normalized):
        return "hip_right"
    if re.fullmatch(r"A[PS]\d+LW[A-Z0-9]*\.PRO", normalized):
        return "forearm_left"
    if re.fullmatch(r"A[PS]\d+RW[A-Z0-9]*\.PRO", normalized):
        return "forearm_right"
    if any(token in normalized for token in ("WB", "TOTAL", "WHOLE")):
        return "total_body"
    return "unsupported"


def _payload(p_file: ApexPFile, *tags: int) -> bytes:
    return next((value for tag in tags if (value := p_file.payload(tag))), b"")


def _year(p_file: ApexPFile) -> str | None:
    for tag in (0x041F, 0x041D):
        match = YEAR_PATTERN.search(decode_text(p_file.payload(tag)))
        if match:
            return match.group(0)
    return None


def _image_flags(protocol: str, images: tuple[ProcessedImage, ...], r_file: ApexRFile) -> list[str]:
    flags: list[str] = []
    tags = {image.tag for image in images}
    if protocol in {"spine_pa", "hip_left", "hip_right"} and 0x0165 not in tags:
        flags.append("missing_secondary_processed_image")
    if protocol == "spine_pa" and r_file.height < 150:
        flags.append("spine_scan_unusually_short")
    if protocol == "unsupported":
        flags.append("unsupported_protocol")
    for image in images:
        clipped = sum(value in (0, 255) for value in image.pixels) / len(image.pixels)
        if clipped > 0.02:
            flags.append(f"processed_image_0x{image.tag:04x}_clipped")
    saturated = sum(value in (0, 65535) for value in r_file.samples) / len(r_file.samples)
    if saturated > 0.001:
        flags.append("raw_signal_saturated")
    return flags


def load_study(pair: ApexPair) -> ApexStudy:
    p_file = parse_p_file(pair.p_path)
    r_file = parse_r_file(pair.r_path)
    protocol_code = decode_text(p_file.payload(0x0150))
    protocol = _protocol(protocol_code)
    patient_key = _payload(p_file, 0x03E9, 0x03EA)
    flags: list[str] = []
    if not patient_key:
        patient_key = bytes.fromhex(p_file.sha256)
        flags.append("missing_patient_group_key")
    study_key = _payload(p_file, 0x041B) or bytes.fromhex(p_file.sha256)
    device_key = _payload(p_file, 0x041C) or b"unknown-device"
    flags.extend(_image_flags(protocol, p_file.images, r_file))
    software = decode_text(_payload(p_file, 0x0428, 0x0429)) or None
    return ApexStudy(
        pair.key,
        p_file,
        r_file,
        protocol,
        Path(protocol_code.replace("\\", "/")).name if protocol_code else "unknown",
        _year(p_file),
        software,
        patient_key,
        study_key,
        device_key,
        tuple(sorted(set(flags))),
    )


def load_dataset(root: Path) -> ApexDataset:
    return ApexDataset(tuple(load_study(pair) for pair in discover_pairs(root)))
