from __future__ import annotations

import hashlib
import re
import struct
import sys
from array import array
from dataclasses import dataclass, field
from pathlib import Path


TLV_HEADER_BYTES = 6
MAX_APEX_FILE_BYTES = 64 * 1024 * 1024
MAX_RECORDS = 10_000
MAX_IMAGE_PIXELS = 25_000_000
RAW_PHASE_COUNT = 6
PROCESSED_IMAGE_TAGS = (0x0046, 0x0165)
SINGLETON_P_TAGS = (0x0150, 0x03E9, 0x03EA, 0x041B, 0x041C, 0x041D, 0x041F, 0x0428, 0x0429)
LOADER_VERSION = "hologic-apex-loader/1.0.0"


class ApexFormatError(ValueError):
    """Raised when a P/R file violates the validated APEX container structure."""


@dataclass(frozen=True)
class TlvRecord:
    offset: int
    tag: int
    total_length: int
    payload: bytes = field(repr=False)


@dataclass(frozen=True)
class ProcessedImage:
    tag: int
    width: int
    height: int
    descriptor: int
    pixels: bytes = field(repr=False)


@dataclass(frozen=True)
class ApexPFile:
    path: Path = field(repr=False)
    sha256: str
    records: tuple[TlvRecord, ...] = field(repr=False)
    images: tuple[ProcessedImage, ...]

    def payload(self, tag: int) -> bytes | None:
        record = next((item for item in self.records if item.tag == tag), None)
        return record.payload if record else None


@dataclass(frozen=True)
class ApexRFile:
    path: Path = field(repr=False)
    sha256: str
    records: tuple[TlvRecord, ...] = field(repr=False)
    sample_width: int
    height: int
    phase_count: int
    samples: array = field(repr=False)

    @property
    def logical_width(self) -> int:
        return self.sample_width // self.phase_count

    def phase(self, index: int) -> array:
        if not 0 <= index < self.phase_count:
            raise IndexError(f"phase must be in [0, {self.phase_count})")
        return array("H", self.samples[index::self.phase_count])


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _read_bounded(path: Path) -> bytes:
    try:
        size = path.stat().st_size
    except OSError as error:
        raise ApexFormatError(f"Unable to stat {path.name}: {error}") from error
    if size <= 0:
        raise ApexFormatError(f"{path.name} is empty")
    if size > MAX_APEX_FILE_BYTES:
        raise ApexFormatError(f"{path.name} exceeds the {MAX_APEX_FILE_BYTES}-byte safety limit")
    try:
        return path.read_bytes()
    except OSError as error:
        raise ApexFormatError(f"Unable to read {path.name}: {error}") from error


def parse_tlv(data: bytes, *, source: str = "APEX file") -> tuple[TlvRecord, ...]:
    records: list[TlvRecord] = []
    offset = 0
    while offset < len(data):
        if len(records) >= MAX_RECORDS:
            raise ApexFormatError(f"{source} contains more than {MAX_RECORDS} records")
        if len(data) - offset < TLV_HEADER_BYTES:
            raise ApexFormatError(f"{source} has {len(data) - offset} trailing byte(s) at offset {offset}")
        tag, total_length = struct.unpack_from("<HI", data, offset)
        if total_length < TLV_HEADER_BYTES:
            raise ApexFormatError(f"{source} has invalid record length {total_length} at offset {offset}")
        end = offset + total_length
        if end > len(data):
            raise ApexFormatError(
                f"{source} record 0x{tag:04X} at offset {offset} ends beyond the file"
            )
        records.append(TlvRecord(offset, tag, total_length, data[offset + TLV_HEADER_BYTES:end]))
        offset = end
    return tuple(records)


def records_by_tag(records: tuple[TlvRecord, ...]) -> dict[int, tuple[TlvRecord, ...]]:
    grouped: dict[int, list[TlvRecord]] = {}
    for record in records:
        grouped.setdefault(record.tag, []).append(record)
    return {tag: tuple(items) for tag, items in grouped.items()}


def decode_text(payload: bytes | None) -> str:
    """Decode APEX text without leaking control characters into logs or manifests."""
    if not payload:
        return ""
    value = payload
    if len(value) % 2 == 0 and value:
        odd_zero_fraction = sum(value[index] == 0 for index in range(1, len(value), 2)) / max(1, len(value) // 2)
        if odd_zero_fraction >= 0.35:
            try:
                decoded = value.decode("utf-16le").rstrip("\x00")
                return re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", decoded).strip()
            except UnicodeDecodeError:
                pass
    trimmed = value.rstrip(b"\x00")
    for encoding in ("ascii", "cp1251"):
        try:
            decoded = trimmed.decode(encoding)
            return re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", decoded).strip()
        except UnicodeDecodeError:
            continue
    return ""


def uint16_value(payload: bytes | None, *, field_name: str) -> int:
    if payload is None or len(payload) != 2:
        actual = "missing" if payload is None else f"{len(payload)} bytes"
        raise ApexFormatError(f"{field_name} must be uint16; got {actual}")
    return struct.unpack("<H", payload)[0]


def _single_payload(grouped: dict[int, tuple[TlvRecord, ...]], tag: int, source: str) -> bytes | None:
    matches = grouped.get(tag, ())
    if len(matches) > 1:
        raise ApexFormatError(f"{source} contains duplicate singleton tag 0x{tag:04X}")
    return matches[0].payload if matches else None


def parse_p_file(path: Path) -> ApexPFile:
    data = _read_bounded(path)
    records = parse_tlv(data, source=path.name)
    grouped = records_by_tag(records)
    for tag in SINGLETON_P_TAGS:
        _single_payload(grouped, tag, path.name)
    images: list[ProcessedImage] = []
    for tag in PROCESSED_IMAGE_TAGS:
        payload = _single_payload(grouped, tag, path.name)
        if payload is None:
            continue
        if len(payload) < 6:
            raise ApexFormatError(f"{path.name} image tag 0x{tag:04X} has no complete header")
        width, height, descriptor = struct.unpack_from("<HHH", payload)
        pixel_count = width * height
        if width <= 0 or height <= 0 or pixel_count > MAX_IMAGE_PIXELS:
            raise ApexFormatError(f"{path.name} image tag 0x{tag:04X} has unsafe dimensions {width}x{height}")
        # Empirically, descriptor is 1 for spine/forearm and 3 for hip, while
        # both layouts store exactly one uint8 sample per pixel. It is not a
        # SamplesPerPixel field and must not be used as a byte multiplier.
        if len(payload) != 6 + pixel_count:
            raise ApexFormatError(
                f"{path.name} image tag 0x{tag:04X} has {len(payload) - 6} pixels; expected {pixel_count}"
            )
        images.append(ProcessedImage(tag, width, height, descriptor, payload[6:]))
    if not any(item.tag == 0x0046 for item in images):
        raise ApexFormatError(f"{path.name} does not contain required processed image tag 0x0046")
    return ApexPFile(path, _sha256(data), records, tuple(images))


def parse_r_file(path: Path) -> ApexRFile:
    data = _read_bounded(path)
    records = parse_tlv(data, source=path.name)
    grouped = records_by_tag(records)
    sample_width = uint16_value(_single_payload(grouped, 0x003A, path.name), field_name="R width (0x003A)")
    height = uint16_value(_single_payload(grouped, 0x003B, path.name), field_name="R height (0x003B)")
    raw = _single_payload(grouped, 0x00CA, path.name)
    if raw is None:
        raise ApexFormatError(f"{path.name} does not contain raw array tag 0x00CA")
    if sample_width <= 0 or height <= 0 or sample_width * height > MAX_IMAGE_PIXELS * RAW_PHASE_COUNT:
        raise ApexFormatError(f"{path.name} has unsafe raw dimensions {sample_width}x{height}")
    if sample_width % RAW_PHASE_COUNT:
        raise ApexFormatError(
            f"{path.name} width {sample_width} is not divisible by the six transmission measurements"
        )
    expected_bytes = sample_width * height * 2
    if len(raw) != expected_bytes:
        raise ApexFormatError(f"{path.name} raw payload has {len(raw)} bytes; expected {expected_bytes}")
    samples = array("H")
    samples.frombytes(raw)
    if sys.byteorder != "little":
        samples.byteswap()
    return ApexRFile(path, _sha256(data), records, sample_width, height, RAW_PHASE_COUNT, samples)
