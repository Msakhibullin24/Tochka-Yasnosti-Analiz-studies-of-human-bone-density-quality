"""Run isolated research candidates on one DXA image; never alters QC output.

The checkpoints are acquired by fetch_dxa_candidates.py. A box is mandatory for
LiteMedSAM: the checkpoint segments the prompted area, it does not locate anatomy.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))


def checked_path(path: Path, digest: str) -> Path:
    if not path.is_file():
        raise FileNotFoundError(f"Missing candidate asset: {path}")
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    if actual != digest:
        raise ValueError(f"Candidate asset checksum mismatch: {path}")
    return path


def load_image(path: Path):
    import numpy as np
    from PIL import Image

    if path.suffix.lower() in {".dcm", ".dicom"}:
        from app.preprocessing import prepare_dicom

        image = prepare_dicom(path.read_bytes()).image
    else:
        image = np.asarray(Image.open(path).convert("L"))
    if image.ndim != 2 or image.size == 0:
        raise ValueError("Expected a nonempty grayscale image")
    return image


def candidate_model(name: str, assets: Path, catalog: dict):
    import torch
    from safetensors.torch import load_file

    weights = catalog["weights"][name]
    path = checked_path(assets / weights["path"], weights["sha256"])
    if name == "lite_medsam":
        source = assets / "sources/litemedsam_source"
        if not source.is_dir():
            raise FileNotFoundError(f"Missing LiteMedSAM source: {source}")
        sys.path.insert(0, str(source))
        from segment_anything.modeling import MaskDecoder, PromptEncoder, TwoWayTransformer
        from tiny_vit_sam import TinyViT

        class LiteMedSAM(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.image_encoder = TinyViT(
                    img_size=256, in_chans=3, embed_dims=[64, 128, 160, 320],
                    depths=[2, 2, 6, 2], num_heads=[2, 4, 5, 10],
                    window_sizes=[7, 7, 14, 7], mlp_ratio=4., drop_rate=0.,
                    drop_path_rate=0., use_checkpoint=False, mbconv_expand_ratio=4.,
                    local_conv_size=3, layer_lr_decay=0.8,
                )
                self.prompt_encoder = PromptEncoder(
                    embed_dim=256, image_embedding_size=(64, 64),
                    input_image_size=(256, 256), mask_in_chans=16,
                )
                self.mask_decoder = MaskDecoder(
                    num_multimask_outputs=3,
                    transformer=TwoWayTransformer(
                        depth=2, embedding_dim=256, mlp_dim=2048, num_heads=8,
                    ), transformer_dim=256, iou_head_depth=3, iou_head_hidden_dim=256,
                )

        model = LiteMedSAM()
        model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True), strict=True)
    else:
        state = load_file(str(path), device="cpu")
        if name == "dax_resnet18_a":
            from torchvision.models import resnet18

            model = resnet18(weights=None)
            model.fc = torch.nn.Identity()
        elif name in {"dax_vit_t16_a", "dax_vit_t8_a"}:
            source = assets / "sources/dax_source/code"
            if not source.is_dir():
                raise FileNotFoundError(f"Missing DAX source: {source}")
            sys.path.insert(0, str(source))
            from vision_transformer_dax import vit_tiny

            model = vit_tiny(patch_size=16 if name == "dax_vit_t16_a" else 8)
        else:
            raise ValueError(f"Unknown candidate: {name}")
        model.load_state_dict(state, strict=True)
    return model.eval()


def segment(model, image, box):
    import cv2
    import numpy as np
    import torch
    import torch.nn.functional as functional

    height, width = image.shape
    x0, y0, x1, y1 = box
    if not (0 <= x0 < x1 <= width and 0 <= y0 < y1 <= height):
        raise ValueError(f"Box must satisfy 0<=x0<x1<={width}, 0<=y0<y1<={height}")
    scale = 256 / max(height, width)
    new_height = max(1, round(height * scale))
    new_width = max(1, round(width * scale))
    rgb = np.repeat(image[:, :, None], 3, axis=2)
    resized = cv2.resize(rgb, (new_width, new_height), interpolation=cv2.INTER_AREA).astype(np.float32)
    resized = (resized - resized.min()) / max(float(resized.max() - resized.min()), 1e-8)
    padded = np.pad(resized, ((0, 256 - new_height), (0, 256 - new_width), (0, 0)))
    tensor = torch.from_numpy(padded).permute(2, 0, 1).unsqueeze(0).float()
    box_tensor = torch.tensor([[[x0 * scale, y0 * scale, x1 * scale, y1 * scale]]], dtype=torch.float32)
    with torch.inference_mode():
        embedding = model.image_encoder(tensor)
        sparse, dense = model.prompt_encoder(points=None, boxes=box_tensor, masks=None)
        logits, predicted_iou = model.mask_decoder(
            image_embeddings=embedding, image_pe=model.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse, dense_prompt_embeddings=dense,
            multimask_output=False,
        )
        logits = functional.interpolate(logits[..., :new_height, :new_width],
                                        size=(height, width), mode="bilinear", align_corners=False)
        mask = (torch.sigmoid(logits)[0, 0] > 0.5).to(torch.uint8).numpy() * 255
    return mask, float(predicted_iou[0, 0])


def features(model, image):
    import numpy as np
    import torch
    from PIL import Image

    # Deliberately uses the project's preview normalization. DAX upstream was
    # trained on 16-bit fluoroscopy, so this is a technical experiment only.
    resized = np.asarray(Image.fromarray(image).resize((224, 224), Image.Resampling.BILINEAR))
    rgb = np.repeat(resized[:, :, None], 3, axis=2).copy()
    tensor = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).float() / 255.0
    with torch.inference_mode():
        vector = model(tensor).flatten().cpu().numpy().astype("float32")
    if not np.isfinite(vector).all():
        raise ValueError("DAX produced non-finite features")
    return vector


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=["lite_medsam", "dax_resnet18_a", "dax_vit_t16_a", "dax_vit_t8_a"], required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--box", nargs=4, type=int, metavar=("X0", "Y0", "X1", "Y1"))
    parser.add_argument("--assets", type=Path, default=ROOT / "data/specialists")
    args = parser.parse_args()
    if args.model == "lite_medsam" and args.box is None:
        parser.error("LiteMedSAM requires --box in original image pixels")
    if args.model != "lite_medsam" and args.box is not None:
        parser.error("--box is only used with LiteMedSAM")
    expected_suffix = ".png" if args.model == "lite_medsam" else ".npy"
    if args.output.suffix.lower() != expected_suffix:
        parser.error(f"--output must end in {expected_suffix} for {args.model}")
    import numpy as np
    from PIL import Image

    catalog = json.loads((ROOT / "docs/competition/dxa_candidate_sources.json").read_text())
    image = load_image(args.input)
    start = perf_counter()
    model = candidate_model(args.model, args.assets, catalog)
    if args.model == "lite_medsam":
        mask, score = segment(model, image, args.box)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(mask).save(args.output)
        result = {"kind": "prompted_mask", "box_xyxy": args.box, "predicted_iou": score,
                  "mask_pixels": int(np.count_nonzero(mask)), "requires_expert_review": True}
    else:
        vector = features(model, image)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        np.save(args.output, vector, allow_pickle=False)
        result = {"kind": "research_features", "features": int(vector.size),
                  "dxa_quality_validated": False, "upstream_preprocessing_matched": False}
    result.update(model=args.model, input=str(args.input), output=str(args.output),
                  elapsed_seconds=round(perf_counter() - start, 3))
    report = args.output.with_suffix(args.output.suffix + ".json")
    report.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
