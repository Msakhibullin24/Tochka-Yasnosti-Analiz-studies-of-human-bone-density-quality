"""Pinned, optional FleXray anatomy masks for research on AP DXA images.

FleXray was trained on radiographs, not DXA. Its output never changes QC labels.
Only the single fp16 ONNX member is downloaded; no ensemble or training data.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from time import perf_counter
from urllib.request import urlopen

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
REVISION = "6b63ddf09cfdff99aea494e3738cf243a4e22cdb"
BASE_URL = f"https://huggingface.co/VictorButoi/flexray/resolve/{REVISION}/members/flux0375/"
ASSETS = {
    "flux0375.onnx": ("onnx/flexray-flux0375-256-fp16.onnx", 203297320,
                       "b8eb8851121cc694f9b529f2cd46f4f1bda29fcbb0eada7b1512af0d57462f15"),
    "config.yml": ("config.yml", 1373,
                   "7c7d52b2404d5e3cb5347036feac994ab8b8a18dd6bef5cfa47b749077af727a"),
    "label_schema.json": ("label_schema.json", 1162,
                          "6f60e2369859b7e17e275c04394c9ec33ac194a8118111577886b1c7fd3cf9f5"),
    "preprocessing.json": ("preprocessing.json", 315,
                           "5994ede07dc359e7fd62215ed1e97b811360fccacb6327bd91b6def439926155"),
}
REGION_LABELS = {
    "spine": ["vertebra_t12", "vertebra_l1", "vertebra_l2", "vertebra_l3", "vertebra_l4", "vertebra_l5"],
    "hip_left": ["femurs", "hips"],
    "hip_right": ["femurs", "hips"],
}


def _digest(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def verified(directory: Path) -> bool:
    return all((directory / name).is_file()
               and (directory / name).stat().st_size == size
               and _digest(directory / name) == digest
               for name, (_, size, digest) in ASSETS.items())


def fetch(directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    for name, (relative, size, digest) in ASSETS.items():
        target = directory / name
        if target.exists():
            if target.stat().st_size != size or _digest(target) != digest:
                raise ValueError(f"Existing FleXray asset differs: {target}")
            continue
        part = directory / (name + ".part")
        try:
            with urlopen(BASE_URL + relative + "?download=true", timeout=60) as response, part.open("wb") as output:
                for block in iter(lambda: response.read(1024 * 1024), b""):
                    output.write(block)
            if part.stat().st_size != size or _digest(part) != digest:
                raise ValueError(f"FleXray download checksum or size mismatch: {name}")
            part.replace(target)
        finally:
            part.unlink(missing_ok=True)


def _prepare(pixels: np.ndarray, metadata: dict) -> np.ndarray:
    """Match upstream array path: uint8 scaling, center pad, bilinear, percentile."""
    from PIL import Image
    if pixels.ndim != 2 or pixels.dtype != np.uint8 or pixels.size == 0:
        raise ValueError("Expected nonempty uint8 grayscale DXA pixels")
    if metadata != {
        "color_mode": "grayscale", "image_size": [256, 256], "mean": None,
        "normalization": {"eps": 1e-8, "percentiles": [0.5, 99.5], "scheme": "percentile_minmax"},
        "pad_to_square": True, "probability_mode": "multilabel", "scale": "zero_one", "std": None,
    }:
        raise ValueError("Unexpected FleXray preprocessing contract")
    height, width = pixels.shape
    side = max(height, width)
    canvas = np.zeros((side, side), np.float32)
    canvas[(side - height) // 2:(side - height) // 2 + height,
           (side - width) // 2:(side - width) // 2 + width] = pixels.astype(np.float32) / 255
    resized = np.asarray(Image.fromarray(canvas).resize((256, 256), Image.Resampling.BILINEAR), dtype=np.float32)
    lo, hi = np.quantile(resized, [0.005, 0.995])
    normalized = (np.clip(resized, lo, hi) - lo) / max(hi - lo, 1e-8)
    return np.ascontiguousarray(normalized[None, None], dtype=np.float32)


def predict(pixels: np.ndarray, region: str, directory: Path, output: Path) -> dict:
    if region not in REGION_LABELS:
        raise ValueError(f"Unsupported FleXray region: {region}")
    if not verified(directory):
        raise FileNotFoundError(f"Missing or corrupt FleXray bundle: {directory}")
    import onnxruntime as ort

    labels = json.loads((directory / "label_schema.json").read_text())["label_names"]
    metadata = json.loads((directory / "preprocessing.json").read_text())
    selected = REGION_LABELS[region]
    if any(label not in labels for label in selected):
        raise ValueError("FleXray label schema lacks required anatomy")
    session = ort.InferenceSession(str(directory / "flux0375.onnx"), providers=["CPUExecutionProvider"])
    if [(item.name, item.shape) for item in session.get_inputs()] != [("input", [1, 1, 256, 256])]:
        raise ValueError("Unexpected FleXray ONNX input contract")
    start = perf_counter()
    result = session.run(None, {"input": _prepare(pixels, metadata)})[0]
    if result.shape != (1, len(labels), 256, 256) or not np.isfinite(result).all():
        raise ValueError("Unexpected or nonfinite FleXray output")
    arrays = {name: result[0, labels.index(name)].astype(np.float32) for name in selected}
    output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output / "flexray_probabilities.npz", **arrays)
    return {
        "status": "research_anatomy_masks", "responsibility": "anatomy_segmentation_only",
        "model": "VictorButoi/flexray/members/flux0375", "revision": REVISION,
        "license": "CC-BY-NC-4.0", "input_source": "project_normalized_dxa_pixels",
        "upstream_dxa_validated": False, "affects_decision": False,
        "mask_space": "padded_256x256", "source_shape_yx": list(pixels.shape),
        "probabilities_file": "flexray_probabilities.npz",
        "labels": {name: {"pixels_above_0_5": int((arr >= 0.5).sum()),
                          "mean_probability": round(float(arr.mean()), 6)} for name, arr in arrays.items()},
        "inference_seconds": round(perf_counter() - start, 3),
        "limitations": "Radiograph model; anatomy masks are not DXA QC labels or verified physical measurements.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["fetch", "verify"])
    parser.add_argument("--directory", type=Path, default=ROOT / "data/specialists/weights/flexray")
    args = parser.parse_args()
    if args.action == "fetch":
        fetch(args.directory)
    ok = verified(args.directory)
    print(json.dumps({"model": "flexray_flux0375_onnx", "verified": ok, "path": str(args.directory)}))
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
