from __future__ import annotations

import hashlib
import io
import json
import os
import re
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any

from PIL import Image

from .annotations import AnnotationDocument


STUDY_ID = re.compile(r"^ST-[A-F0-9]{20}$")
ASSET_NAME = re.compile(r"^p_[0-9a-f]{4}\.png$")
MAX_MANIFEST_BYTES = 128 * 1024 * 1024


class DatasetUnavailable(RuntimeError):
    pass


class DatasetContractError(ValueError):
    pass


def _require_child(root: Path, candidate: Path) -> Path:
    resolved = candidate.resolve()
    try:
        resolved.relative_to(root.resolve())
    except ValueError as error:
        raise DatasetContractError("Resolved path escapes the configured dataset root") from error
    return resolved


class DatasetRepository:
    def __init__(self, root: Path | None, annotation_root: Path | None = None):
        self.root = root.expanduser().resolve() if root else None
        self.annotation_root = (
            annotation_root.expanduser().resolve()
            if annotation_root
            else self.root / "annotations" if self.root else None
        )

    def _available_root(self) -> Path:
        if self.root is None:
            raise DatasetUnavailable("OSSEO_DATASET_ROOT is not configured")
        manifest = self.root / "manifest.jsonl"
        if not manifest.is_file():
            raise DatasetUnavailable(f"Dataset manifest is not available at {manifest}")
        if manifest.stat().st_size > MAX_MANIFEST_BYTES:
            raise DatasetContractError("Dataset manifest exceeds the safety limit")
        return self.root

    def records(self) -> list[dict[str, Any]]:
        root = self._available_root()
        records: list[dict[str, Any]] = []
        seen: set[str] = set()
        with (root / "manifest.jsonl").open(encoding="utf-8") as stream:
            for index, line in enumerate(stream, 1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as error:
                    raise DatasetContractError(f"Invalid JSON at manifest line {index}") from error
                if not isinstance(record, dict):
                    raise DatasetContractError(f"Manifest line {index} must contain an object")
                study_id = str(record.get("studyId", ""))
                if not STUDY_ID.fullmatch(study_id) or study_id in seen:
                    raise DatasetContractError(f"Invalid or duplicate studyId at manifest line {index}")
                privacy = record.get("privacy")
                if not isinstance(privacy, dict) or privacy.get("directIdentifiersExported") is not False:
                    raise DatasetContractError(f"Study {study_id} is not marked as de-identified")
                processed = record.get("processedImages", [])
                if not isinstance(processed, list):
                    raise DatasetContractError(f"Study {study_id} has invalid processedImages")
                for image in processed:
                    if not isinstance(image, dict):
                        raise DatasetContractError(f"Study {study_id} has an invalid processed image entry")
                    tag = str(image.get("tag", "")).removeprefix("0x").lower()
                    if not re.fullmatch(r"[0-9a-f]{4}", tag):
                        raise DatasetContractError(f"Study {study_id} has an invalid processed image tag")
                raw = record.get("raw", {})
                if not isinstance(raw, dict):
                    raise DatasetContractError(f"Study {study_id} has invalid raw metadata")
                try:
                    transmission_count = int(raw.get("transmissionCount", 0))
                except (TypeError, ValueError) as error:
                    raise DatasetContractError(f"Study {study_id} has invalid transmissionCount") from error
                if not 0 <= transmission_count <= 6:
                    raise DatasetContractError(f"Study {study_id} has invalid transmissionCount")
                seen.add(study_id)
                records.append(record)
        return records

    def _annotation_paths(self, study_id: str) -> list[Path]:
        self._validate_study_id(study_id)
        if self.annotation_root is None or not self.annotation_root.is_dir():
            return []
        return sorted(self.annotation_root.glob(f"{study_id}--*.json"))

    def annotations(self, study_id: str) -> list[dict[str, Any]]:
        documents: list[dict[str, Any]] = []
        for path in self._annotation_paths(study_id):
            try:
                documents.append(json.loads(path.read_text(encoding="utf-8")))
            except (OSError, json.JSONDecodeError) as error:
                raise DatasetContractError(f"Invalid annotation file: {path.name}") from error
        return sorted(
            documents,
            key=lambda item: str(item.get("expert", {}).get("createdAt", "")),
            reverse=True,
        )

    def _decorate(self, record: dict[str, Any]) -> dict[str, Any]:
        study_id = str(record["studyId"])
        result = dict(record)
        result["assets"] = [
            {
                "name": f"p_{str(image['tag']).removeprefix('0x').lower()}.png",
                "tag": image["tag"],
                "url": f"/api/v1/datasets/current/studies/{study_id}/assets/p_{str(image['tag']).removeprefix('0x').lower()}.png",
            }
            for image in record.get("processedImages", [])
        ]
        result["rawChannels"] = [
            {
                "index": index,
                "label": f"Канал {index}",
                "url": f"/api/v1/datasets/current/studies/{study_id}/raw/{index}.png",
            }
            for index in range(int(record.get("raw", {}).get("transmissionCount", 0)))
        ]
        result["annotationCount"] = len(self._annotation_paths(study_id))
        return result

    def list_studies(
        self,
        *,
        protocol: str | None = None,
        split: str | None = None,
        review_required: bool | None = None,
        annotated: bool | None = None,
        query: str = "",
    ) -> list[dict[str, Any]]:
        query = query.strip().lower()
        selected: list[dict[str, Any]] = []
        for record in self.records():
            study_id = str(record["studyId"])
            has_annotation = bool(self._annotation_paths(study_id))
            if protocol and record.get("protocol") != protocol:
                continue
            if split and record.get("split") != split:
                continue
            if review_required is not None and bool(record.get("quality", {}).get("reviewRequired")) != review_required:
                continue
            if annotated is not None and has_annotation != annotated:
                continue
            if query and query not in " ".join((study_id, str(record.get("patientGroupId", "")), str(record.get("protocolCode", "")))).lower():
                continue
            selected.append(self._decorate(record))
        return selected

    def get_study(self, study_id: str) -> dict[str, Any]:
        self._validate_study_id(study_id)
        record = next((item for item in self.records() if item["studyId"] == study_id), None)
        if record is None:
            raise KeyError(study_id)
        result = self._decorate(record)
        result["annotations"] = self.annotations(study_id)
        return result

    def summary(self) -> dict[str, Any]:
        root = self._available_root()
        records = self.records()
        source_summary: dict[str, Any] = {}
        summary_path = root / "dataset_summary.json"
        if summary_path.is_file():
            source_summary = json.loads(summary_path.read_text(encoding="utf-8"))
        annotation_counts = {record["studyId"]: len(self._annotation_paths(record["studyId"])) for record in records}
        return {
            **source_summary,
            "studyCount": len(records),
            "protocols": dict(sorted(Counter(str(item.get("protocol", "unknown")) for item in records).items())),
            "splits": dict(sorted(Counter(str(item.get("split", "unassigned")) for item in records).items())),
            "annotatedStudyCount": sum(count > 0 for count in annotation_counts.values()),
            "annotationCount": sum(annotation_counts.values()),
            "reviewRequiredCount": sum(bool(item.get("quality", {}).get("reviewRequired")) for item in records),
            "phaseSemantics": "unverified",
        }

    def asset_path(self, study_id: str, asset_name: str) -> Path:
        root = self._available_root()
        self._validate_study_id(study_id)
        if not ASSET_NAME.fullmatch(asset_name):
            raise DatasetContractError("Unsupported processed asset name")
        record = next((item for item in self.records() if item["studyId"] == study_id), None)
        if record is None:
            raise FileNotFoundError(study_id)
        declared = {
            f"p_{str(image.get('tag', '')).removeprefix('0x').lower()}.png"
            for image in record.get("processedImages", [])
            if isinstance(image, dict)
        }
        if asset_name not in declared:
            raise FileNotFoundError(asset_name)
        path = _require_child(root, root / "processed" / study_id / asset_name)
        if not path.is_file():
            raise FileNotFoundError(path)
        return path

    def raw_channel_png(self, study_id: str, channel: int) -> bytes:
        import numpy as np

        root = self._available_root()
        self._validate_study_id(study_id)
        record = next((item for item in self.records() if item["studyId"] == study_id), None)
        if record is None:
            raise FileNotFoundError(study_id)
        raw = record.get("raw", {})
        transmission_count = int(raw.get("transmissionCount", 0))
        if channel not in range(transmission_count):
            raise DatasetContractError(f"Raw channel must be in the range 0..{max(transmission_count - 1, 0)}")
        path = _require_child(root, root / "raw" / study_id / "transmissions.npy")
        if not path.is_file():
            raise FileNotFoundError(path)
        cube = np.load(path, mmap_mode="r", allow_pickle=False)
        expected_shape = tuple(int(value) for value in raw.get("shape", []))
        if cube.size > 36_000_000 or cube.ndim != 3 or cube.shape[2] != transmission_count or cube.dtype != np.uint16:
            raise DatasetContractError("Raw cube violates the manifest uint16 contract")
        if expected_shape and tuple(cube.shape) != expected_shape:
            raise DatasetContractError("Raw cube dimensions differ from the manifest")
        values = np.asarray(cube[:, :, channel], dtype=np.float32)
        low, high = np.percentile(values, (1, 99))
        if high <= low:
            normalized = np.zeros(values.shape, dtype=np.uint8)
        else:
            normalized = (np.clip((values - low) / (high - low), 0, 1) * 255).round().astype(np.uint8)
        image = Image.fromarray(normalized, mode="L")
        image.thumbnail((1024, 1024))
        output = io.BytesIO()
        image.save(output, format="PNG", optimize=True)
        return output.getvalue()

    def save_annotation(self, document: AnnotationDocument) -> dict[str, Any]:
        record = self.get_study(document.studyId)
        if document.patientGroupId != record.get("patientGroupId"):
            raise DatasetContractError("Annotation patientGroupId does not match the dataset manifest")
        if document.deviceGroup and document.deviceGroup != record.get("deviceGroup"):
            raise DatasetContractError("Annotation deviceGroup does not match the dataset manifest")
        expected_protocol = record.get("protocol")
        annotation_protocol = "unsupported" if expected_protocol in {"forearm_left", "forearm_right", "unsupported"} else expected_protocol
        if document.protocol != annotation_protocol:
            raise DatasetContractError("Annotation protocol does not match the dataset manifest")
        if self.annotation_root is None:
            raise DatasetUnavailable("Annotation storage is not configured")
        self.annotation_root.mkdir(parents=True, exist_ok=True, mode=0o700)
        reader_hash = hashlib.sha256(document.expert.readerId.encode("utf-8")).hexdigest()[:12].upper()
        target = self.annotation_root / f"{document.studyId}--{reader_hash}--{document.expert.readIndex:02d}.json"
        payload = document.model_dump(mode="json")
        encoded = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=self.annotation_root, delete=False) as stream:
            stream.write(encoded)
            temporary = Path(stream.name)
        temporary.chmod(0o600)
        os.replace(temporary, target)
        return payload

    def annotation_jsonl(self) -> bytes:
        rows = [
            json.dumps(annotation, ensure_ascii=False, separators=(",", ":"))
            for record in self.records()
            for annotation in self.annotations(str(record["studyId"]))
        ]
        return (("\n".join(rows) + "\n") if rows else "").encode("utf-8")

    def coco_export(self) -> dict[str, Any]:
        """Build a de-identified COCO-style geometry export from expert reads."""
        records = self.records()
        documents = {
            str(record["studyId"]): self.annotations(str(record["studyId"]))
            for record in records
        }
        region_names = sorted({
            str(region["name"])
            for annotations in documents.values()
            for document in annotations
            for region in document.get("regions", [])
        })
        landmark_names = sorted({
            str(landmark["name"])
            for annotations in documents.values()
            for document in annotations
            for landmark in document.get("landmarks", [])
        })
        category_ids = {name: index + 1 for index, name in enumerate(region_names)}
        landmark_category_id = len(category_ids) + 1
        categories = [
            {"id": category_id, "name": name, "supercategory": "expert_region"}
            for name, category_id in category_ids.items()
        ]
        if landmark_names:
            categories.append({
                "id": landmark_category_id,
                "name": "anatomical_landmarks",
                "supercategory": "expert_keypoints",
                "keypoints": landmark_names,
                "skeleton": [],
            })

        images: list[dict[str, Any]] = []
        annotations: list[dict[str, Any]] = []
        annotation_id = 1
        for image_id, record in enumerate(records, 1):
            processed = record.get("processedImages", [])
            reference = processed[0] if processed else {}
            width = int(reference.get("width", 0))
            height = int(reference.get("height", 0))
            tag = str(reference.get("tag", "0x0046")).removeprefix("0x").lower()
            images.append({
                "id": image_id,
                "file_name": f"processed/{record['studyId']}/p_{tag}.png",
                "width": width,
                "height": height,
                "study_id": record["studyId"],
                "patient_group_id": record["patientGroupId"],
                "split": record.get("split", "unassigned"),
            })
            for document in documents[str(record["studyId"])]:
                expert = document.get("expert", {})
                attributes = {
                    "reader_id": expert.get("readerId"),
                    "read_index": expert.get("readIndex"),
                    "adjudicated": expert.get("adjudicated", False),
                    "overall_action": document.get("overallAction"),
                    "evaluable": document.get("evaluable", True),
                }
                for region in document.get("regions", []):
                    points = region.get("points", [])
                    if len(points) < 2 or width <= 0 or height <= 0:
                        continue
                    pixel_points = [[float(x) * width, float(y) * height] for x, y in points]
                    xs = [point[0] for point in pixel_points]
                    ys = [point[1] for point in pixel_points]
                    bbox = [min(xs), min(ys), max(xs) - min(xs), max(ys) - min(ys)]
                    annotations.append({
                        "id": annotation_id,
                        "image_id": image_id,
                        "category_id": category_ids[str(region["name"])],
                        "bbox": bbox,
                        "area": bbox[2] * bbox[3],
                        "segmentation": [[coordinate for point in pixel_points for coordinate in point]] if region.get("geometryType") == "polygon" else [],
                        "iscrowd": 0,
                        "attributes": {**attributes, "geometry_type": region.get("geometryType")},
                    })
                    annotation_id += 1
                if landmark_names and width > 0 and height > 0:
                    by_name = {str(item["name"]): item for item in document.get("landmarks", [])}
                    keypoints: list[float | int] = []
                    visible_count = 0
                    for name in landmark_names:
                        point = by_name.get(name)
                        visible = bool(point and point.get("visible", True))
                        keypoints.extend([
                            float(point.get("x", 0)) * width if point else 0,
                            float(point.get("y", 0)) * height if point else 0,
                            2 if visible else 0,
                        ])
                        visible_count += int(visible)
                    annotations.append({
                        "id": annotation_id,
                        "image_id": image_id,
                        "category_id": landmark_category_id,
                        "keypoints": keypoints,
                        "num_keypoints": visible_count,
                        "bbox": [0, 0, width, height],
                        "area": width * height,
                        "iscrowd": 0,
                        "attributes": attributes,
                    })
                    annotation_id += 1
        return {
            "info": {"description": "Osseo AI de-identified expert annotations", "version": "1.0.0"},
            "images": images,
            "annotations": annotations,
            "categories": categories,
        }

    @staticmethod
    def _validate_study_id(study_id: str) -> None:
        if not STUDY_ID.fullmatch(study_id):
            raise DatasetContractError("Invalid studyId")
