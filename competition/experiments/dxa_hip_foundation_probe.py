"""Exploratory, study-held-out hip QC probe for frozen foundation encoders.

Run with the research environment (Transformers 5) and a local Hugging Face
snapshot. The organiser's combined hip label is not an anatomical gold standard.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import AutoImageProcessor, AutoModel

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
from dxaqc.dicom_io import read_any  # noqa: E402
from source_integrity import inspect_sources  # noqa: E402
from dxa_hip_vit_qc_cv import evaluate  # noqa: E402

SUPPORTED = {
    "google/medsiglip-448": "siglip",
    "facebook/dinov3-vith16plus-pretrain-lvd1689m": "dinov3_vit",
    "facebook/dinov3-vit7b16-pretrain-lvd1689m": "dinov3_vit",
}


def digest(path: Path) -> str:
    hash_ = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            hash_.update(block)
    return hash_.hexdigest()


def encode(model: torch.nn.Module, pixel_values: torch.Tensor, kind: str) -> np.ndarray:
    parameter = next(model.parameters())
    pixel_values = pixel_values.to(device=parameter.device, dtype=parameter.dtype)
    with torch.inference_mode():
        if kind == "siglip":
            output = model.get_image_features(pixel_values=pixel_values)
            if not isinstance(output, torch.Tensor):
                output = output.pooler_output
        else:
            output = model(pixel_values=pixel_values).last_hidden_state[:, 0]
    vector = output[0].float().cpu().numpy()
    if vector.ndim != 1 or not np.isfinite(vector).all():
        raise ValueError("Encoder returned an invalid image embedding")
    return vector


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--model-dir", required=True, type=Path)
    parser.add_argument("--repo", required=True, choices=SUPPORTED)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--cache-dir", type=Path, default=HERE.parent / "data/specialists/feature_cache")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--smoke", action="store_true", help="Encode one audited hip DXA and report time only")
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("Choose a new report path")
    model_dir = args.model_dir.resolve()
    weights = sorted(model_dir.glob("*.safetensors"))
    if not weights:
        raise FileNotFoundError("Complete safetensors checkpoint is required")
    if (model_dir / "model.safetensors.index.json").is_file():
        shard_index = json.loads((model_dir / "model.safetensors.index.json").read_text())
        expected = set(shard_index["weight_map"].values())
        if {p.name for p in weights} != expected:
            raise ValueError("Sharded checkpoint is incomplete")
    config = json.loads((model_dir / "config.json").read_text())
    kind = SUPPORTED[args.repo]
    if config["model_type"] != kind:
        raise ValueError("Model kind and repository do not match")
    weight_sha = hashlib.sha256(json.dumps(
        [(p.name, digest(p)) for p in weights]).encode()).hexdigest()
    metadata = model_dir / ".cache/huggingface/download" / (weights[0].name + ".metadata")
    revision = metadata.read_text().splitlines()[0] if metadata.exists() else "unrecorded"

    labels_path = HERE / "labels/image_labels.csv"
    labels = pd.read_csv(labels_path)
    labels = labels[labels.quality_class.notna()].reset_index(drop=True)
    audit = inspect_sources([("organizer", labels_path, args.dataset)])
    audited = {row["path"]: row["pixel_sha256_current"] for row in audit["entries"]}
    fingerprint = hashlib.sha256(json.dumps(
        [(row["path"], row["pixel_sha256_current"]) for row in audit["entries"]],
        ensure_ascii=False).encode()).hexdigest()
    baseline_path = HERE / "reports/hip_rotation_ablation.json"
    baseline = json.loads(baseline_path.read_text())
    if baseline["labels_sha256"] != digest(labels_path) or baseline["source_fingerprint"] != fingerprint:
        raise ValueError("Existing baseline does not match the current DXA cohort")
    baseline_oof = pd.read_csv(HERE / "reports/hip_rotation_ablation_oof.csv")

    selected = np.flatnonzero(labels.region.isin(["hip_left", "hip_right"]).to_numpy())
    cache = args.cache_dir / args.repo.replace("/", "_") / f"{weight_sha[:16]}_{fingerprint[:16]}"
    cache.mkdir(parents=True, exist_ok=True)
    paths = {int(i): cache / f"{int(i):03d}_{audited[labels.first_source_path.iloc[i]]}.npy" for i in selected}
    todo = selected[:1] if args.smoke else selected
    missing = [int(i) for i in todo if not paths[int(i)].is_file()]
    if missing:
        torch.set_num_threads(4)
        device = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
        if device == "auto":
            device = "cpu"
        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable")
        if device == "cpu":
            available = 0
            meminfo = Path("/proc/meminfo")
            if meminfo.exists():
                for line in meminfo.read_text().splitlines():
                    if line.startswith("MemAvailable:"):
                        available = int(line.split()[1]) * 1024
                        break
            if available and sum(p.stat().st_size for p in weights) > 0.7 * available:
                raise RuntimeError("Checkpoint is too large for available CPU memory; run on a GPU workstation")
        dtype = torch.float32 if device == "cpu" else (
            torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16)
        processor = AutoImageProcessor.from_pretrained(model_dir, local_files_only=True)
        model = AutoModel.from_pretrained(model_dir, local_files_only=True, dtype=dtype).to(device).eval()
        for serial, i in enumerate(missing, 1):
            row = labels.iloc[i]
            image = read_any(args.dataset / row.first_source_path, getattr(row, "pixel_mm", None))
            if image.pixel_sha256 != audited[row.first_source_path]:
                raise ValueError("Audited DXA pixels changed during extraction")
            pixels = image.pixels[:, ::-1] if row.region == "hip_left" else image.pixels
            rgb = np.repeat(np.ascontiguousarray(pixels)[:, :, None], 3, axis=2)
            values = processor(images=rgb, return_tensors="pt")["pixel_values"]
            vector = encode(model, values, kind)
            temporary = paths[i].with_suffix(".tmp.npy")
            np.save(temporary, vector)
            temporary.replace(paths[i])
            if serial == 1 or serial % 10 == 0 or serial == len(missing):
                print(f"Encoded {serial}/{len(missing)} new hip DXA; embedding {vector.size}", flush=True)
        del model
    vectors = [np.load(paths[int(i)], allow_pickle=False) for i in todo]
    if len({v.shape for v in vectors}) != 1 or not all(np.isfinite(v).all() for v in vectors):
        raise ValueError("Feature cache contains invalid embeddings")
    if args.smoke:
        print(json.dumps({"repo": args.repo, "model_revision": revision,
                          "model_weight_sha256": weight_sha, "encoded_images": 1,
                          "embedding_dim": int(vectors[0].size)}))
        return

    features = np.full((len(labels), vectors[0].size), np.nan, dtype=np.float32)
    features[selected] = np.stack(vectors)
    variants, oof = evaluate(labels, features, baseline_oof)
    variants["foundation_lr"] = variants.pop("vit_cls_lr")
    oof = oof.rename(columns={"vit_cls_lr_score": "foundation_lr_score",
                              "vit_cls_lr_pred": "foundation_lr_pred"})
    report = {"scope": "Research-only organizer DXA combined hip positioning/rotation label",
              "protocol": "Frozen model image embedding, fixed balanced C=.01 logistic probe; same study-held-out outer and inner folds as the existing baseline",
              "images": len(oof), "studies": int(oof.study.nunique()),
              "positives": int(oof.target.sum()), "source_fingerprint": fingerprint,
              "labels_sha256": digest(labels_path), "model_url": f"https://huggingface.co/{args.repo}",
              "model_revision": revision, "model_weight_sha256": weight_sha,
              "image_processor_sha256": digest(model_dir / "preprocessor_config.json"),
              "code_sha256": digest(Path(__file__)), "variants": variants,
              "clinical_validation": False, "model_affects_decision": False,
              "limitations": ["Combined hip label does not identify anatomical rotation subtype.",
                              "Organiser images were inspected before; no independent external test.",
                              "Known region is supplied and patient identity across studies is unverified.",
                              "No expert landmark or mask labels for clinical/anatomical accuracy."]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    oof.to_csv(args.output.with_name(args.output.stem + "_oof.csv"), index=False)
    print(json.dumps(variants, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
