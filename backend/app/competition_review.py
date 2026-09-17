"""Validate expert mapping and compile candidate image labels, without freezing splits."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path
import shutil
import tempfile

from .competition_data import REGIONS, write_json, write_jsonl

CRITERIA = tuple(dict.fromkeys(name for _, fields in REGIONS.values() for name in fields))
STATUSES = {"pending", "mapped", "reviewed", "unassessable"}


def jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def unique_index(rows, field):
    result = {}
    for row in rows:
        key = row[field]
        if not key or key in result:
            raise ValueError(f"Missing or duplicate {field}")
        result[key] = row
    return result


def compile_review(images, studies, reviews):
    images = unique_index(images, "image_id")
    studies = unique_index(studies, "source_study_key")
    reviews = unique_index(reviews, "image_id")
    if set(images) != set(reviews):
        raise ValueError("Review must contain exactly the manifest image IDs; rows are missing or unknown")
    issues, candidates, counts = [], [], Counter()
    assignments = defaultdict(list)

    def issue(identifier, code):
        issues.append({"image_id": identifier, "code": code})

    for identifier, image in images.items():
        row = reviews[identifier]
        study = studies[image["source_study_key"]]
        for field, expected in (("image_path", image["image_path"]), ("sheet_path", image["sheet_path"]),
                                ("excel_row", str(study["excel_row"]))):
            if row.get(field) != expected:
                raise ValueError(f"Immutable field changed for {identifier}: {field}")
        status, region = row.get("review_status"), row.get("anatomical_region", "")
        if status not in STATUSES or region not in {"", "spine", "hip_right", "hip_left", "unsupported", "uncertain"}:
            raise ValueError(f"Invalid status or region for {identifier}")
        values = {}
        for field in ("quality_class", *CRITERIA):
            value = row.get(field)
            if value not in ("", "0", "1"):
                raise ValueError(f"Invalid binary value for {identifier}: {field}")
            values[field] = int(value) if value else None
        counts[status] += 1
        if status == "pending":
            issue(identifier, "pending_review")
            continue
        if not row.get("reviewer", "").strip():
            raise ValueError(f"Reviewer code required for {identifier}")
        if status == "unassessable" or region in {"unsupported", "uncertain"}:
            if status != "unassessable" or not row.get("notes", "").strip():
                raise ValueError(f"Unassessable images require status and reason: {identifier}")
            if any(value is not None for value in values.values()):
                raise ValueError(f"Unassessable image cannot carry clinical labels: {identifier}")
            issue(identifier, "explicitly_unassessable")
            continue
        if region not in REGIONS:
            raise ValueError(f"Mapped/reviewed image requires a supported region: {identifier}")
        applicable = set(REGIONS[region][1])
        if any(values[name] is not None for name in set(CRITERIA) - applicable):
            raise ValueError(f"Non-applicable criterion filled for {identifier}")
        original = study["regions"][region]
        assignments[(image["source_study_key"], region)].append(identifier)
        if status == "mapped":
            if any(value is not None for value in values.values()):
                raise ValueError(f"Use reviewed status when supplying clinical labels: {identifier}")
            quality, criteria = original["quality_class"], dict(original["criteria"])
            origin = "organizer_labels_after_region_mapping"
            known = [v for v in criteria.values() if v is not None]
            if quality is not None and known and ((quality == 0 and 1 in known) or
                    (quality == 1 and len(known) == len(criteria) and not any(known))):
                issue(identifier, "original_overall_criteria_disagreement_requires_review")
                continue
        else:
            # Blank expert fields stay unknown; never refill them from the old Excel.
            quality = values["quality_class"]
            criteria = {name: values[name] for name in original["criteria"]}
            origin = "expert_review"
            known = [v for v in criteria.values() if v is not None]
            changed = quality != original["quality_class"] or criteria != original["criteria"]
            incomplete = quality is None or len(known) != len(criteria)
            disagrees = quality is not None and known and ((quality == 0 and 1 in known) or
                        (quality == 1 and len(known) == len(criteria) and not any(known)))
            if (changed or incomplete or disagrees) and not row.get("notes", "").strip():
                raise ValueError(f"Changed, incomplete or disagreeing labels require notes: {identifier}")
        if quality is None:
            issue(identifier, "unknown_overall_quality")
        candidates.append({
            "image_id": identifier, "image_path": image["image_path"],
            "pixel_sha256": image["pixel_sha256"], "source_study_key": image["source_study_key"],
            "anatomical_region": region, "quality_class": quality, "criteria": criteria,
            "label_origin": origin, "reviewer": row["reviewer"], "review_status": status,
            "notes": row.get("notes", ""), "original_labels": original,
            "split_status": "not_frozen", "excel_row": study["excel_row"],
        })
    ambiguous = {identifier for ids in assignments.values() if len(ids) > 1 for identifier in ids}
    for identifier in sorted(ambiguous):
        issue(identifier, "multiple_distinct_images_for_one_study_region")
    candidates = [row for row in candidates if row["image_id"] not in ambiguous]
    summary = {
        "status": "candidate_labels_not_training_release", "image_count": len(images),
        "review_status_counts": dict(counts), "candidate_images": len(candidates),
        "binary_labeled_images": sum(row["quality_class"] is not None for row in candidates),
        "clinical_labels_independently_reviewed": sum(row["label_origin"] == "expert_review" for row in candidates),
        "issue_counts": dict(Counter(issue["code"] for issue in issues)),
        "remaining_release_steps": ["Confirm label definitions and unresolved clinical issues",
                                    "Review related-study candidates and freeze grouped train/validation partitions",
                                    "Verify physical scale and ROI limitations against organizer requirements"],
    }
    return summary, candidates, issues


def run(data: Path, review: Path, output: Path):
    if output.exists():
        raise FileExistsError("Refusing to overwrite an existing review compilation")
    with review.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        required = {"image_id", "image_path", "sheet_path", "excel_row", "anatomical_region",
                    "quality_class", *CRITERIA, "reviewer", "review_status", "notes"}
        if not reader.fieldnames or len(reader.fieldnames) != len(set(reader.fieldnames)) or required - set(reader.fieldnames):
            raise ValueError("Missing or duplicate review columns")
        reviews = list(reader)
    summary, candidates, issues = compile_review(jsonl(data / "manifest.jsonl"), jsonl(data / "study_labels.jsonl"), reviews)
    summary["input_sha256"] = {name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in
                               (("manifest", data / "manifest.jsonl"), ("study_labels", data / "study_labels.jsonl"), ("review", review))}
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".review-", dir=output.parent))
    try:
        write_json(staging / "summary.json", summary)
        write_jsonl(staging / "candidate_image_labels.jsonl", candidates)
        write_jsonl(staging / "issues.jsonl", issues)
        staging.rename(output)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--review", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.data, args.review, args.output), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
