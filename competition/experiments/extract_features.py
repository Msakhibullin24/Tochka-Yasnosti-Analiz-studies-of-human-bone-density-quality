"""Extract frozen ResNet18 embeddings (local ImageNet weights, no network) for reviewed images.

Usage: python extract_features.py <img_size>
Left hips are mirrored so both hips share one orientation (femur on the image left).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torchvision
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
IMAGES = ROOT / "data" / "competition-v1" / "images"
LABELS = ROOT / "competition" / "labels" / "image_labels.csv"
WEIGHTS = ROOT / "data" / "baseline-assets" / "resnet18-f37072fd.pth"
OUT = ROOT / "data" / "runs" / "features"
MEAN = np.array([0.485, 0.456, 0.406])[:, None, None]
STD = np.array([0.229, 0.224, 0.225])[:, None, None]


def load_image(sha: str, region: str) -> np.ndarray:
    path = next(IMAGES.glob(f"*-IMG-{sha[:20]}.png"))
    a = np.asarray(Image.open(path).convert("L"))
    return a[:, ::-1] if region == "hip_left" else a


def letterbox(a: np.ndarray, size: int) -> np.ndarray:
    h, w = a.shape
    s = size / max(h, w)
    im = Image.fromarray(np.ascontiguousarray(a)).resize((max(1, round(w * s)), max(1, round(h * s))), Image.BICUBIC)
    canvas = Image.new("L", (size, size), 0)
    canvas.paste(im, ((size - im.width) // 2, (size - im.height) // 2))
    return np.asarray(canvas)


def build() -> torch.nn.Module:
    m = torchvision.models.resnet18()
    m.load_state_dict(torch.load(WEIGHTS, map_location="cpu"))
    return m.eval()


def embed(m: torch.nn.Module, x: torch.Tensor) -> torch.Tensor:
    x = m.maxpool(m.relu(m.bn1(m.conv1(x))))
    outs = []
    for layer in (m.layer1, m.layer2, m.layer3, m.layer4):
        x = layer(x)
        outs += [x.mean((2, 3)), x.amax((2, 3))]
    return torch.cat(outs[2:], 1)  # layer2..layer4, avg+max


def main() -> None:
    size = int(sys.argv[1])
    torch.set_num_threads(10)
    df = pd.read_csv(LABELS)
    m = build()
    feats = []
    with torch.inference_mode():
        for r in df.itertuples():
            a = letterbox(load_image(r.pixel_sha256, r.region), size).astype(np.float32) / 255.0
            x = ((np.repeat(a[None], 3, 0) - MEAN) / STD).astype(np.float32)
            feats.append(embed(m, torch.from_numpy(x[None]))[0].numpy())
    OUT.mkdir(parents=True, exist_ok=True)
    np.save(OUT / f"resnet18_{size}.npy", np.stack(feats))
    print("saved", np.stack(feats).shape)


if __name__ == "__main__":
    main()
