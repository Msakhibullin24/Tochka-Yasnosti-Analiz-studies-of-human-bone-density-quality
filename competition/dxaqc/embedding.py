"""Frozen ResNet18 embedding (local weights, CPU, deterministic)."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np

WEIGHTS = Path(__file__).resolve().parents[1] / "weights" / "resnet18-f37072fd.pth"
WEIGHTS_SHA256 = "f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec"
SIZE = 320
_MEAN = np.array([0.485, 0.456, 0.406], np.float32)[:, None, None]
_STD = np.array([0.229, 0.224, 0.225], np.float32)[:, None, None]


def letterbox(img: np.ndarray, size: int = SIZE) -> np.ndarray:
    h, w = img.shape
    s = size / max(h, w)
    nh, nw = max(1, round(h * s)), max(1, round(w * s))
    r = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_CUBIC)
    out = np.zeros((size, size), np.uint8)
    y0, x0 = (size - nh) // 2, (size - nw) // 2
    out[y0:y0 + nh, x0:x0 + nw] = r
    return out


@lru_cache(maxsize=1)
def _model():
    import torch
    import torchvision

    torch.manual_seed(0)
    m = torchvision.models.resnet18()
    m.load_state_dict(torch.load(WEIGHTS, map_location="cpu"))
    return m.eval()


def embed(img: np.ndarray) -> np.ndarray:
    """1792-d vector: avg+max pooled layer2..layer4 activations."""
    import torch

    m = _model()
    x = letterbox(img).astype(np.float32) / 255.0
    x = torch.from_numpy(((np.repeat(x[None], 3, 0) - _MEAN) / _STD)[None])
    with torch.inference_mode():
        x = m.maxpool(m.relu(m.bn1(m.conv1(x))))
        x = m.layer1(x)
        outs = []
        for layer in (m.layer2, m.layer3, m.layer4):
            x = layer(x)
            outs += [x.mean((2, 3)), x.amax((2, 3))]
    return torch.cat(outs, 1)[0].numpy().astype(np.float32)
