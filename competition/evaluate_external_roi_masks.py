"""Evaluate prompted LiteMedSAM against Ramathibodi AP DXA ROI masks.

The prompt is derived from the reference mask: this is an oracle-box
segmentation benchmark, never an automatic landmark or QC test.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from dxaqc.dicom_io import read_dxa
from run_dxa_candidate import ROOT, candidate_model, segment


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_seg_nrrd(path: Path, image_shape: tuple[int, int]) -> dict[str, np.ndarray]:
    """Read the narrow gzip/uint8/4D Segmentation NRRD contract in this source."""
    raw = path.read_bytes()
    header, separator, payload = raw.partition(b"\n\n")
    if not separator:
        raise ValueError("NRRD header is incomplete")
    lines = header.decode("ascii").splitlines()
    if not lines or lines[0] != "NRRD0005":
        raise ValueError("Unsupported NRRD version")
    fields, segments = {}, {}
    for line in lines[1:]:
        if line.startswith("#") or not line:
            continue
        if ":=" in line:
            key, value = line.split(":=", 1)
            if key.startswith("Segment"):
                segments[key] = value
        elif ": " in line:
            key, value = line.split(": ", 1)
            fields[key] = value
    if (fields.get("type") != "uint8" or fields.get("dimension") != "4"
            or fields.get("encoding") != "gzip"
            or fields.get("kinds") != "list domain domain domain"):
        raise ValueError("Unsupported NRRD encoding or dimension")
    sizes = tuple(int(item) for item in fields["sizes"].split())
    if len(sizes) != 4 or sizes[1:3] != image_shape[::-1] or sizes[3] != 1:
        raise ValueError("NRRD raster does not match DICOM image")
    decoded = gzip.decompress(payload)
    if len(decoded) != int(np.prod(sizes)):
        raise ValueError("NRRD decoded byte count mismatch")
    raster = np.frombuffer(decoded, dtype=np.uint8).reshape(sizes, order="F")
    names = [key for key in segments if key.endswith("_ID")]
    if not names:
        raise ValueError("NRRD contains no named segments")
    masks = {}
    for key in names:
        prefix = key[:-3]
        name = segments[key]
        layer = int(segments[prefix + "_Layer"])
        label = int(segments[prefix + "_LabelValue"])
        if name in masks or not 0 <= layer < sizes[0] or not 1 <= label <= 255:
            raise ValueError("Duplicate or invalid NRRD segment")
        mask = np.asarray(raster[layer, :, :, 0].T == label)
        masks[name] = mask
    return masks


def bbox(mask: np.ndarray, padding: int = 4) -> tuple[int, int, int, int]:
    ys, xs = np.where(mask)
    if not len(xs):
        raise ValueError("Empty reference mask")
    return (max(0, int(xs.min()) - padding), max(0, int(ys.min()) - padding),
            min(mask.shape[1], int(xs.max()) + padding + 1),
            min(mask.shape[0], int(ys.max()) + padding + 1))


def overlap(reference: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    if reference.shape != predicted.shape:
        raise ValueError("Mask dimensions differ")
    first, second = reference.astype(bool), predicted.astype(bool)
    intersection = int(np.count_nonzero(first & second))
    a, b = int(np.count_nonzero(first)), int(np.count_nonzero(second))
    return {"dice": 2 * intersection / (a + b) if a + b else 1.0,
            "iou": intersection / (a + b - intersection) if a + b - intersection else 1.0,
            "reference_pixels": a, "predicted_pixels": b}


def patient_bootstrap(cases: list[dict], repeats: int = 2000) -> dict:
    grouped = {}
    for case in cases:
        grouped.setdefault(case["patient_sha256"], []).append(case)
    units = list(grouped.values())
    rng = np.random.default_rng(17)
    dice, delta = [], []
    for _ in range(repeats):
        sample = [case for i in rng.integers(len(units), size=len(units)) for case in units[i]]
        dice.append(float(np.mean([case["lite_medsam"]["dice"] for case in sample])))
        delta.append(float(np.mean([case["lite_medsam"]["dice"] -
                                    case["filled_box_baseline"]["dice"] for case in sample])))
    return {"mean_dice_95": [float(np.percentile(dice, 2.5)), float(np.percentile(dice, 97.5))],
            "delta_dice_vs_filled_box_95": [float(np.percentile(delta, 2.5)),
                                             float(np.percentile(delta, 97.5))]}


def run(root: Path, assets: Path, catalog_path: Path) -> dict:
    import torch

    torch.set_num_threads(4)
    catalog = json.loads(catalog_path.read_text())
    model = candidate_model("lite_medsam", assets, catalog)
    annotations = root / "Annotation"
    patient_dirs = sorted(path for path in annotations.iterdir() if path.is_dir())
    if not patient_dirs:
        raise ValueError("No Ramathibodi Annotation patients")
    cases = []
    empty_references = []
    seen_images = set()
    for patient in patient_dirs:
        for region in ("spine", "hip"):
            image_path = patient / "images" / f"{region}_image.dcm"
            mask_path = patient / "segmentations" / f"{region}_image.seg.nrrd"
            image = read_dxa(image_path).pixels
            image_hash = sha256(image_path)
            if image_hash in seen_images:
                raise ValueError("Duplicate DICOM in Annotation partition")
            seen_images.add(image_hash)
            masks = read_seg_nrrd(mask_path, image.shape)
            for name, reference in masks.items():
                if not name.endswith("_bone_area"):
                    continue
                if not reference.any():
                    empty_references.append({"patient_sha256": hashlib.sha256(patient.name.encode()).hexdigest(),
                                             "region": region, "segment": name})
                    continue
                prompt = bbox(reference)
                started = perf_counter()
                predicted, predicted_iou = segment(model, image, prompt)
                duration = perf_counter() - started
                rectangle = np.zeros(image.shape, dtype=bool)
                x0, y0, x1, y1 = prompt
                rectangle[y0:y1, x0:x1] = True
                cases.append({"patient_sha256": hashlib.sha256(patient.name.encode()).hexdigest(),
                              "region": region, "segment": name, "image_sha256": image_hash,
                              "reference_sha256": sha256(mask_path), "oracle_box_xyxy": list(prompt),
                              "lite_medsam": overlap(reference, predicted),
                              "filled_box_baseline": overlap(reference, rectangle),
                              "model_predicted_iou": predicted_iou,
                              "inference_seconds": round(duration, 4)})
    summary = {}
    for region in ("spine", "hip"):
        selected = [case for case in cases if case["region"] == region]
        summary[region] = {"masks": len(selected),
                           "patients": len({case["patient_sha256"] for case in selected}),
                           "mean_dice": float(np.mean([case["lite_medsam"]["dice"] for case in selected])),
                           "median_dice": float(np.median([case["lite_medsam"]["dice"] for case in selected])),
                           "mean_box_dice": float(np.mean([case["filled_box_baseline"]["dice"] for case in selected])),
                           "mean_iou": float(np.mean([case["lite_medsam"]["iou"] for case in selected])),
                           "patient_bootstrap": patient_bootstrap(selected)}
    by_segment = {}
    for name in sorted({case["segment"] for case in cases}):
        selected = [case for case in cases if case["segment"] == name]
        by_segment[name] = {"masks": len(selected),
                            "mean_dice": float(np.mean([case["lite_medsam"]["dice"] for case in selected])),
                            "mean_box_dice": float(np.mean([case["filled_box_baseline"]["dice"] for case in selected]))}
    return {"protocol": "Ramathibodi BMD Annotation only; reference-mask-derived box with 4-pixel padding; bone-area ROI masks; one model load; CPU 4 threads",
            "limitations": ["Oracle prompt from reference mask; not autonomous detection or competition QC.",
                            "Source segmentations are vendor ROI bone areas, not independently reviewed anatomical contours.",
                            "Ten Hologic patients with burned-in ROI graphics; no external AP GE validation.",
                            "Mask coordinate correspondence is checked by raster size, not visually attested for every case."],
            "model_sha256": catalog["weights"]["lite_medsam"]["sha256"],
            "model_affects_decision": False, "patients": len(patient_dirs),
            "summary": summary, "by_segment": by_segment,
            "empty_reference_segments": empty_references, "cases": cases}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT / "data/external/ramathibodi/extracted/Osteoporosis/BMD")
    parser.add_argument("--assets", type=Path, default=ROOT / "data/specialists")
    parser.add_argument("--catalog", type=Path, default=ROOT / "docs/competition/dxa_candidate_sources.json")
    parser.add_argument("--output", type=Path, default=ROOT / "competition/reports/external_ramathibodi_roi_masks_2026_09_24.json")
    args = parser.parse_args()
    result = run(args.root, args.assets, args.catalog)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"patients": result["patients"], "summary": result["summary"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
