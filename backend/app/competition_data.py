"""Prepare the organizer's GE DICOM/XLSX release for local expert review.

This is a data audit, not an anatomical classifier or a de-identification tool.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
from pathlib import Path
import shutil
import tempfile
import warnings
import xml.etree.ElementTree as ET
import zipfile
from collections import Counter, defaultdict

import numpy as np
import pydicom
from PIL import Image, ImageDraw

from .evidence import _difference_hash

VERSION = "competition-data/1.0.0"
REGIONS = {
    "spine": ("J", {"spine_coverage": "C", "spine_axis": "D", "spine_artifact": "E"}),
    "hip_right": ("K", {"hip_position_rotation": "F", "hip_roi_coverage": "G"}),
    "hip_left": ("L", {"hip_position_rotation": "H", "hip_roi_coverage": "I"}),
}
NS = {"s": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}


def digest(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, records: list[dict]) -> None:
    path.write_text("".join(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n" for r in records), encoding="utf-8")


def read_labels(path: Path) -> list[dict]:
    # ponytail: only the organizer's fixed worksheet contract; use a general XLSX
    # library if subsequent releases introduce different layouts or calculations.
    with zipfile.ZipFile(path) as archive:
        strings = []
        if "xl/sharedStrings.xml" in archive.namelist():
            strings = ["".join(t.text or "" for t in item.findall(".//s:t", NS))
                       for item in ET.fromstring(archive.read("xl/sharedStrings.xml"))]
        workbook = ET.fromstring(archive.read("xl/workbook.xml"))
        sheets = workbook.findall("s:sheets/s:sheet", NS)
        sheet = next((s for s in sheets if s.get("name") == "Калибровка"), None)
        if sheet is None:
            raise ValueError("Expected worksheet Калибровка")
        rel_id = sheet.get("{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id")
        relationships = ET.fromstring(archive.read("xl/_rels/workbook.xml.rels"))
        target = next(r.attrib["Target"] for r in relationships if r.get("Id") == rel_id)
        member = target.lstrip("/") if target.startswith("/") else "xl/" + target
        raw_rows = {}
        for row in ET.fromstring(archive.read(member)).findall(".//s:sheetData/s:row", NS):
            cells = {}
            for cell in row:
                ref = cell.get("r", "")
                col = "".join(c for c in ref if c.isalpha())
                value = cell.find("s:v", NS)
                if cell.find("s:f", NS) is not None and (value is None or value.text is None):
                    raise ValueError(f"Formula has no cached value: {ref}")
                text = (value.text or "") if value is not None else ""
                if cell.get("t") == "s":
                    text = strings[int(text)]
                elif cell.get("t") == "inlineStr":
                    text = "".join(t.text or "" for t in cell.findall(".//s:t", NS))
                cells[col] = text.strip()
            raw_rows[int(row.attrib["r"])] = cells
    if raw_rows.get(1, {}).get("B") != "study":
        raise ValueError("Expected study in B1")
    expected = {"C": "корректная укладка", "D": "правильно выравнена ось позвоночника",
                "E": "наличие посторонних предметов", "F": "позиционирование/ротация",
                "G": "корректности области интересов", "H": "позиционирование/ротация",
                "I": "корректности области интересов", "J": "Позвоночник",
                "K": "Проксимальный отдел правого бедра", "L": "Проксимальный отдел левого бедра"}
    if any(not raw_rows.get(2, {}).get(c, "").startswith(v) for c, v in expected.items()):
        raise ValueError("Worksheet columns changed; review label mapping")
    result, seen = [], set()
    for number, row in sorted(raw_rows.items()):
        if number <= 2 or not row.get("B"):
            continue
        if row["B"] in seen:
            raise ValueError(f"Duplicate study key at row {number}")
        seen.add(row["B"])
        values = {}
        for col in "CDEFGHIJKL":
            value = row.get(col, "")
            if value not in ("", "0", "1"):
                raise ValueError(f"Non-binary label at {col}{number}")
            values[col] = int(value) if value else None
        regions = {region: {"quality_class": values[overall],
                            "criteria": {name: values[col] for name, col in criteria.items()}}
                   for region, (overall, criteria) in REGIONS.items()}
        result.append({"excel_row": number, "source_study_key": row["B"],
                       "regions": regions, "comment": row.get("M", "")})
    if not result:
        raise ValueError("No study labels found")
    return result


def pixel_fingerprint(array: np.ndarray, photometric: str) -> str:
    header = json.dumps([list(array.shape), array.dtype.str, photometric]).encode()
    return digest(header + b"\0" + np.ascontiguousarray(array).tobytes())


def prepare(source: Path, labels_path: Path, output: Path, *, seed: str = "dxa-v1", folds: int = 5) -> dict:
    source, labels_path, output = source.resolve(), labels_path.resolve(), output.resolve()
    if not source.is_dir():
        raise ValueError("Source must be the extracted Исследования directory")
    if output == source or source in output.parents:
        raise ValueError("Output must be outside source")
    if output.exists():
        raise FileExistsError("Refusing to overwrite an existing preparation")
    if folds < 2:
        raise ValueError("At least two folds are required")
    labels = read_labels(labels_path)
    label_by_key = {r["source_study_key"]: r for r in labels}
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".competition-", dir=output.parent))
    try:
        (staging / "images").mkdir()
        (staging / "sheets").mkdir()
        images, sources, issues = {}, [], []
        study_uids, patient_ids = defaultdict(set), set()
        files = sorted(p for p in source.rglob("*") if p.is_file() and p.suffix.lower() in (".dcm", ".dicom"))
        if not files:
            raise ValueError("No DICOM files found")
        for path in files:
            relative = path.relative_to(source)
            if source not in path.resolve().parents or len(relative.parts) < 2:
                raise ValueError("Expected in-root study subdirectories, not loose files or external symlinks")
            key = relative.parts[0]
            record = {"path": relative.as_posix(), "source_study_key": key,
                      "file_sha256": digest(path.read_bytes())}
            try:
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always")
                    ds = pydicom.dcmread(path)
                    list(ds.iterall())  # Materialize lazy tags while warnings are captured.
                    array = ds.pixel_array
                    uid = str(ds.get("StudyInstanceUID", ""))
                    sop = str(ds.get("SOPInstanceUID", ""))
                    uid_valid = pydicom.uid.UID(uid).is_valid
                photo = str(ds.get("PhotometricInterpretation", ""))
                if array.ndim != 2 or array.dtype != np.uint8 or photo not in ("MONOCHROME1", "MONOCHROME2"):
                    raise ValueError("This preparation supports only single-frame 8-bit monochrome release images")
                if not uid or not sop:
                    raise ValueError("Missing study/image UID")
                fingerprint = pixel_fingerprint(array, photo)
                image_id = "IMG-" + fingerprint[:20]
                identity = (key, fingerprint)
                # Keep occurrences in different studies separate, and report their
                # shared content instead of silently combining different labels.
                if identity not in images:
                    asset_id = "ST-" + digest(key.encode())[:12] + "-" + image_id
                    preview = array if photo == "MONOCHROME2" else 255 - array
                    Image.fromarray(preview).save(staging / "images" / f"{asset_id}.png")
                    images[identity] = {"image_id": asset_id, "source_study_key": key,
                        "pixel_sha256": fingerprint, "image_path": f"images/{asset_id}.png",
                        "rows": int(array.shape[0]), "columns": int(array.shape[1]),
                        "anatomical_region": None, "quality_class": None,
                        "label_status": "pending_image_region_review", "source_paths": []}
                record.update({"status": "Success", "image_id": images[identity]["image_id"],
                    "study_uid": uid, "image_uid": sop,
                    "series_uid": str(ds.get("SeriesInstanceUID", "")),
                    "pixel_spacing": [str(v) for v in ds.get("PixelSpacing", [])],
                    "manufacturer": str(ds.get("Manufacturer", "")),
                    "model": str(ds.get("ManufacturerModelName", "")),
                    "uid_syntax_valid": uid_valid,
                    "reader_warning_count": len(caught)})
                images[identity]["source_paths"].append(record["path"])
                study_uids[key].add(uid)
                patient_ids.add(str(ds.get("PatientID", "")))
            except Exception as error:
                record.update({"status": "Failure", "error_type": type(error).__name__})
                issues.append({"type": "decode_failure", "source_study_key": key, "path": record["path"],
                               "error_type": type(error).__name__})
            sources.append(record)
        by_study = defaultdict(list)
        for item in images.values():
            by_study[item["source_study_key"]].append(item)
        for key in sorted({r["source_study_key"] for r in sources} | set(label_by_key)):
            label = label_by_key.get(key)
            if label is None:
                issues.append({"type": "missing_study_labels", "source_study_key": key})
                continue
            label["dicom_study_uids"] = sorted(study_uids[key])
            if len(study_uids[key]) != 1:
                issues.append({"type": "study_uid_cardinality", "source_study_key": key,
                               "uid_count": len(study_uids[key])})
            filled = sum(v["quality_class"] is not None for v in label["regions"].values())
            if filled != len(by_study[key]):
                issues.append({"type": "image_label_count_mismatch", "source_study_key": key,
                               "excel_row": label["excel_row"], "images": len(by_study[key]), "labels": filled})
            for region, data in label["regions"].items():
                values = list(data["criteria"].values())
                if data["quality_class"] is not None and all(v is not None for v in values) and data["quality_class"] != max(values):
                    issues.append({"type": "overall_criteria_disagreement", "source_study_key": key,
                                   "excel_row": label["excel_row"], "region": region,
                                   "quality_class": data["quality_class"], "criteria": data["criteria"]})
        content_groups = defaultdict(set)
        for item in images.values():
            content_groups[item["pixel_sha256"]].add(item["source_study_key"])
        for fingerprint, keys in content_groups.items():
            if len(keys) > 1:
                issues.append({"type": "cross_study_exact_duplicate", "pixel_sha256": fingerprint,
                               "source_study_keys": sorted(keys)})
        # Candidate folds only: no claim of patient independence or rare-label balance.
        assignments, strata = {}, defaultdict(list)
        for label in labels:
            present = [v["quality_class"] for v in label["regions"].values() if v["quality_class"] is not None]
            strata[max(present) if present else "unknown"].append(label["source_study_key"])
        for keys in strata.values():
            for index, key in enumerate(sorted(keys, key=lambda k: digest((seed + "\0" + k).encode()))):
                assignments[key] = index % folds
        for key, items in sorted(by_study.items()):
            sheet = Image.new("L", (340 * len(items), 450), 220)
            draw = ImageDraw.Draw(sheet)
            for index, item in enumerate(items):
                with Image.open(staging / item["image_path"]) as im:
                    im.thumbnail((330, 400))
                    sheet.paste(im, (340 * index + 5, 40))
                draw.text((340 * index + 5, 5), item["image_id"].split("-IMG-")[-1], fill=0)
                item["candidate_fold"] = assignments.get(key)
                item["sheet_path"] = f"sheets/{digest(key.encode())[:12]}.png"
            sheet.save(staging / items[0]["sheet_path"])
        records = sorted(images.values(), key=lambda r: (r["source_study_key"], r["image_id"]))
        perceptual = [(r, _difference_hash(staging / r["image_path"])) for r in records]
        near = []
        (staging / "near_sheets").mkdir()
        # ponytail: O(n²) is small for this 252-image release; use an indexed search
        # when a substantially larger dataset is supplied.
        for (first, a), (second, b) in itertools.combinations(perceptual, 2):
            if first["source_study_key"] != second["source_study_key"] and (a ^ b).bit_count() <= 3:
                sheet_path = f"near_sheets/pair-{len(near) + 1:03}.png"
                sheet = Image.new("L", (680, 450), 220)
                draw = ImageDraw.Draw(sheet)
                for index, item in enumerate((first, second)):
                    with Image.open(staging / item["image_path"]) as im:
                        im.thumbnail((330, 400))
                        sheet.paste(im, (340 * index + 5, 40))
                    draw.text((340 * index + 5, 5), item["image_id"].split("-IMG-")[-1], fill=0)
                sheet.save(staging / sheet_path)
                near.append({"first_image": first["image_id"], "second_image": second["image_id"],
                             "sheet_path": sheet_path, "hamming_distance": (a ^ b).bit_count(),
                             "status": "review_candidate_not_confirmed_duplicate"})
        with (staging / "near_duplicate_review.csv").open("w", encoding="utf-8-sig", newline="") as stream:
            fields = ["first_image", "second_image", "sheet_path", "hamming_distance", "status", "reviewer", "decision", "notes"]
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(near)
        with (staging / "priority_review.csv").open("w", encoding="utf-8-sig", newline="") as stream:
            fields = ["issue_type", "excel_row", "region", "details", "sheet_path", "reviewer", "decision", "notes"]
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            for issue in issues:
                key = issue.get("source_study_key", "")
                writer.writerow({"issue_type": issue["type"], "excel_row": issue.get("excel_row", ""),
                    "region": issue.get("region", ""),
                    "details": json.dumps({k: v for k, v in issue.items() if k not in ("source_study_key", "path")}, ensure_ascii=False),
                    "sheet_path": by_study[key][0]["sheet_path"] if by_study[key] else ""})
        write_jsonl(staging / "manifest.jsonl", records)
        write_jsonl(staging / "sources.jsonl", sources)
        write_jsonl(staging / "study_labels.jsonl", labels)
        write_jsonl(staging / "issues.jsonl", issues)
        write_jsonl(staging / "near_duplicate_candidates.jsonl", near)
        write_json(staging / "candidate_folds.json", {"status": "draft_not_frozen", "seed": seed,
                   "fold_count": folds, "group_unit": "study_folder_not_verified_patient",
                   "stratification": "study_overall_only", "assignments": assignments})
        with (staging / "image_review.csv").open("w", encoding="utf-8-sig", newline="") as stream:
            fields = ["image_id", "image_path", "sheet_path", "excel_row", "anatomical_region",
                      "quality_class", "spine_coverage", "spine_axis", "spine_artifact",
                      "hip_position_rotation", "hip_roi_coverage", "reviewer", "review_status", "notes"]
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            for r in records:
                writer.writerow({"image_id": r["image_id"], "image_path": r["image_path"], "sheet_path": r["sheet_path"],
                                 "excel_row": label_by_key.get(r["source_study_key"], {}).get("excel_row", ""),
                                 "review_status": "pending"})
        manifest_sha = digest((staging / "manifest.jsonl").read_bytes())
        labels_sha = digest(labels_path.read_bytes())
        sources_sha = digest((staging / "sources.jsonl").read_bytes())
        summary = {"version": VERSION,
            "dataset_version": "DXA-" + digest((VERSION + manifest_sha + labels_sha + sources_sha).encode())[:16],
            "labels_sha256": labels_sha, "manifest_sha256": manifest_sha,
            "sources_sha256": sources_sha,
            "source_file_count": len(sources), "decoded_count": sum(r["status"] == "Success" for r in sources),
            "unique_image_count": len(records), "unique_pixel_count": len(content_groups),
            "study_label_count": len(labels), "study_folder_count": len(by_study),
            "duplicate_extra_files": sum(len(r["source_paths"]) - 1 for r in records),
            "issue_counts": dict(Counter(r["type"] for r in issues)), "near_duplicate_candidate_count": len(near),
            "patient_id_distinct_count": len(patient_ids), "patient_independence_verified": False,
            "missing_pixel_spacing_files": sum(not r.get("pixel_spacing") for r in sources if r["status"] == "Success"),
            "invalid_study_uid_files": sum(not r.get("uid_syntax_valid") for r in sources if r["status"] == "Success"),
            "training_ready": False, "image_region_reviews_pending": len(records),
            "label_polarity": "1_means_violation_pending_organizer_confirmation",
            "region_counts": {region: dict(Counter(str(r["regions"][region]["quality_class"]) for r in labels)) for region in REGIONS}}
        write_json(staging / "summary.json", summary)
        for p in staging.rglob("*"):
            p.chmod(0o700 if p.is_dir() else 0o600)
        if output.exists():
            raise FileExistsError("Output appeared during preparation")
        staging.rename(output)
        return summary
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", default="dxa-v1")
    parser.add_argument("--folds", type=int, default=5)
    args = parser.parse_args()
    print(json.dumps(prepare(args.source, args.labels, args.output, seed=args.seed, folds=args.folds), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
