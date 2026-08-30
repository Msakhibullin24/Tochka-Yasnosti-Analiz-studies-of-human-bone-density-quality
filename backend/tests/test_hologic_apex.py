from __future__ import annotations

import json
import struct
from pathlib import Path

import numpy as np
import pytest
from jsonschema import Draft202012Validator

from app.hologic_apex.exporter import export_dataset, study_manifest
from app.hologic_apex.archive import materialize_source
from app.hologic_apex.loader import discover_pairs, load_dataset, pseudonymize
from app.hologic_apex.parser import ApexFormatError, parse_p_file, parse_r_file, parse_tlv


def tlv(tag: int, payload: bytes) -> bytes:
    return struct.pack("<HI", tag, len(payload) + 6) + payload


def image_payload(width: int, height: int, descriptor: int, start: int = 0) -> bytes:
    pixels = bytes((start + index) % 256 for index in range(width * height))
    return struct.pack("<HHH", width, height, descriptor) + pixels


def make_pair(root: Path, *, protocol: str = r"C:\QDR\scanpro\AS8LH10X.PRO") -> tuple[Path, Path]:
    p_path = root / "PX26001A.p01"
    r_path = root / "PX26001A.r01"
    p_path.write_bytes(b"".join((
        tlv(0x03E9, b"PRIVATE-PATIENT"),
        tlv(0x041B, b"PRIVATE-STUDY"),
        tlv(0x041C, b"PRIVATE-DEVICE"),
        tlv(0x041F, b"20260101"),
        tlv(0x0428, "13.3.0.1".encode("utf-16le")),
        tlv(0x0150, protocol.encode("ascii")),
        tlv(0x0046, image_payload(3, 2, 3)),
        tlv(0x0165, image_payload(3, 2, 3, 10)),
    )))
    samples = tuple(range(100, 124))
    r_path.write_bytes(b"".join((
        tlv(0x003A, struct.pack("<H", 12)),
        tlv(0x003B, struct.pack("<H", 2)),
        tlv(0x00CA, struct.pack("<" + "H" * len(samples), *samples)),
    )))
    return p_path, r_path


def test_tlv_parser_rejects_truncated_records() -> None:
    with pytest.raises(ApexFormatError, match="ends beyond the file"):
        parse_tlv(struct.pack("<HI", 0x46, 100) + b"short")


def test_pseudonymization_rejects_weak_keys() -> None:
    with pytest.raises(ValueError, match="at least 16 bytes"):
        pseudonymize(b"too-short", "patient", b"patient")


def test_p_image_descriptor_is_not_treated_as_channel_count(tmp_path: Path) -> None:
    p_path, _ = make_pair(tmp_path)
    parsed = parse_p_file(p_path)
    assert len(parsed.images) == 2
    assert parsed.images[0].descriptor == 3
    assert parsed.images[0].pixels == bytes(range(6))


def test_p_file_rejects_duplicate_identity_records(tmp_path: Path) -> None:
    path = tmp_path / "duplicate.p01"
    path.write_bytes(b"".join((
        tlv(0x03E9, b"FIRST"),
        tlv(0x03E9, b"SECOND"),
        tlv(0x0046, image_payload(3, 2, 1)),
    )))
    with pytest.raises(ApexFormatError, match="duplicate singleton tag 0x03E9"):
        parse_p_file(path)


def test_r_file_is_six_transmissions_per_spatial_point(tmp_path: Path) -> None:
    _, r_path = make_pair(tmp_path)
    parsed = parse_r_file(r_path)
    assert parsed.logical_width == 2
    assert parsed.height == 2
    assert parsed.phase_count == 6
    assert list(parsed.phase(0)) == [100, 106, 112, 118]
    assert list(parsed.phase(5)) == [105, 111, 117, 123]


def test_r_file_rejects_width_that_is_not_divisible_by_six(tmp_path: Path) -> None:
    path = tmp_path / "invalid.r01"
    path.write_bytes(b"".join((
        tlv(0x003A, struct.pack("<H", 10)),
        tlv(0x003B, struct.pack("<H", 1)),
        tlv(0x00CA, struct.pack("<" + "H" * 10, *range(10))),
    )))
    with pytest.raises(ApexFormatError, match="six transmission"):
        parse_r_file(path)


def test_pair_discovery_rejects_incomplete_studies(tmp_path: Path) -> None:
    (tmp_path / "orphan.p01").write_bytes(b"not-read-yet")
    with pytest.raises(ApexFormatError, match="Unpaired"):
        discover_pairs(tmp_path)


def test_directory_with_multiple_archives_requires_explicit_selection(tmp_path: Path) -> None:
    (tmp_path / "one.rar").write_bytes(b"placeholder")
    (tmp_path / "two.rar").write_bytes(b"placeholder")
    with pytest.raises(ApexFormatError, match="select one archive explicitly"):
        with materialize_source(tmp_path):
            pass


def test_manifest_is_keyed_and_contains_no_direct_identifiers(tmp_path: Path) -> None:
    make_pair(tmp_path)
    dataset = load_dataset(tmp_path)
    manifest = study_manifest(dataset.studies[0], b"a sufficiently long test key")
    encoded = json.dumps(manifest)
    assert manifest["loaderVersion"] == "hologic-apex-loader/1.0.0"
    assert manifest["protocol"] == "hip_left"
    assert manifest["raw"]["shape"] == [2, 2, 6]
    assert manifest["raw"]["phaseSemantics"] == "unverified"
    assert "PRIVATE-PATIENT" not in encoded
    assert "PRIVATE-STUDY" not in encoded
    assert "PRIVATE-DEVICE" not in encoded
    schema_path = Path(__file__).resolve().parents[2] / "docs" / "hologic-apex-manifest.schema.json"
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    Draft202012Validator.check_schema(schema)
    Draft202012Validator(schema).validate(manifest)


def test_export_is_atomic_non_overwriting_and_preserves_uint16_cube(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    make_pair(source)
    dataset = load_dataset(source)
    output = tmp_path / "export"
    summary = export_dataset(dataset, output, b"a sufficiently long test key")
    assert summary["exportedStudyCount"] == 1
    manifest = json.loads((output / "manifest.jsonl").read_text(encoding="utf-8"))
    study_id = manifest["studyId"]
    cube = np.load(output / "raw" / study_id / "transmissions.npy", allow_pickle=False)
    assert cube.shape == (2, 2, 6)
    assert cube.dtype == np.uint16
    assert cube[0, 0].tolist() == [100, 101, 102, 103, 104, 105]
    assert (output / "processed" / study_id / "p_0046.png").is_file()
    assert json.loads((output / "splits.json").read_text(encoding="utf-8"))["assignments"][study_id] == "train"
    qc_report = json.loads((output / "qc_report.json").read_text(encoding="utf-8"))
    assert qc_report["studyCount"] == 1
    assert manifest["technicalQc"]["baselineVersion"] == "osseo-technical-qc/1.0.0"
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        export_dataset(dataset, output, b"a sufficiently long test key")
