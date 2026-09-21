"""CPU-only, study-held-out fine-tuning experiment for DXA image quality.

Run in the pinned competition container; this never changes the shipped model.
The ImageNet ResNet18 stem through layer3 is frozen. Layer4 and a binary head
are trained separately for spine and hip, using only images in each outer fold.
No augmentation changes anatomy or labels. The test folds never select epochs,
thresholds, architecture, or hyperparameters.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from torch import nn

from dxaqc.dicom_io import read_any
from dxaqc.embedding import WEIGHTS, WEIGHTS_SHA256, letterbox
from dxaqc.model import group_of

SEED = 17
SIZE = 224
MEAN = torch.tensor([.485, .456, .406], dtype=torch.float32)[None, :, None, None]
STD = torch.tensor([.229, .224, .225], dtype=torch.float32)[None, :, None, None]


def valid_labels(path: Path) -> pd.DataFrame:
    labels = pd.read_csv(path)
    labels = labels[labels.quality_class.notna()].reset_index(drop=True)
    required = {"first_source_path", "pixel_sha256", "study_key", "region", "quality_class"}
    if not required.issubset(labels.columns):
        raise ValueError(f"Missing label columns: {sorted(required - set(labels.columns))}")
    if not labels.region.isin(("spine", "hip_right", "hip_left")).all():
        raise ValueError("Only spine and labelled hip regions are supported")
    if not labels.quality_class.isin((0, 1)).all():
        raise ValueError("Quality labels must be 0 or 1")
    if labels.groupby("pixel_sha256").study_key.nunique().gt(1).any():
        raise ValueError("An identical image occurs in multiple study groups")
    return labels


def image_tensor(image: np.ndarray) -> torch.Tensor:
    plane = torch.from_numpy(letterbox(image, SIZE).copy()).float().div_(255)
    return ((plane[None].repeat(1, 3, 1, 1) - MEAN) / STD).squeeze(0)


def frozen_features(dataset: Path, labels: pd.DataFrame) -> tuple[torch.Tensor, int]:
    import torchvision

    encoder = torchvision.models.resnet18()
    encoder.load_state_dict(torch.load(WEIGHTS, map_location="cpu", weights_only=False))
    encoder.eval()
    result = []
    changed_hashes = 0
    current_hash_study: dict[str, str] = {}
    with torch.inference_mode():
        for i, row in enumerate(labels.itertuples(), 1):
            img = read_any(dataset / row.first_source_path)
            if img.pixels.shape != (row.rows, row.columns):
                raise ValueError(f"Image dimensions changed: {row.first_source_path}")
            prior_study = current_hash_study.setdefault(img.pixel_sha256, row.study_key)
            if prior_study != row.study_key:
                raise ValueError(f"Current pixel duplicate crosses study groups: {row.first_source_path}")
            # Historical labels hash an older pixel normalisation. Keep their stable
            # identifiers for pairing reports, but record a changed current hash.
            changed_hashes += img.pixel_sha256 != row.pixel_sha256
            px = img.pixels[:, ::-1].copy() if row.region == "hip_left" else img.pixels
            x = image_tensor(px).unsqueeze(0)
            x = encoder.maxpool(encoder.relu(encoder.bn1(encoder.conv1(x))))
            x = encoder.layer3(encoder.layer2(encoder.layer1(x)))
            result.append(x.squeeze(0).clone())
            if i % 50 == 0:
                print(f"Prepared {i}/{len(labels)} DICOM images", flush=True)
    return torch.stack(result), changed_hashes


class QualityHead(nn.Module):
    def __init__(self, pretrained_layer4: nn.Module):
        super().__init__()
        import copy
        self.layer4 = copy.deepcopy(pretrained_layer4)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(512, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(self.pool(self.layer4(x)).flatten(1)).flatten()


def train_head(features: torch.Tensor, targets: np.ndarray, train_idx: np.ndarray,
               pretrained_layer4: nn.Module, epochs: int, seed: int) -> QualityHead:
    torch.manual_seed(seed)
    model = QualityHead(pretrained_layer4).train()
    pos = int(targets[train_idx].sum())
    neg = len(train_idx) - pos
    if not pos or not neg:
        raise ValueError("Each training fold must contain both quality classes")
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(neg / pos))
    optimizer = torch.optim.AdamW([
        {"params": model.layer4.parameters(), "lr": 1e-4},
        {"params": model.fc.parameters(), "lr": 1e-3},
    ], weight_decay=1e-3)
    rng = np.random.default_rng(seed)
    for epoch in range(epochs):
        order = rng.permutation(train_idx)
        total = 0.0
        for batch in np.array_split(order, max(1, int(np.ceil(len(order) / 16)))):
            x = features[batch]
            y = torch.from_numpy(targets[batch].astype(np.float32))
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(x), y)
            loss.backward()
            optimizer.step()
            total += float(loss.detach()) * len(batch)
        print(f"  epoch {epoch + 1}/{epochs} train loss {total / len(train_idx):.4f}", flush=True)
    return model.eval()


def metrics(y: np.ndarray, score: np.ndarray, threshold: float = .5) -> dict:
    pred = score >= threshold
    tp = int(((y == 1) & pred).sum())
    fp = int(((y == 0) & pred).sum())
    fn = int(((y == 1) & ~pred).sum())
    tn = int(((y == 0) & ~pred).sum())
    return {"n": len(y), "positives": int(y.sum()), "roc_auc": float(roc_auc_score(y, score)),
            "average_precision": float(average_precision_score(y, score)),
            "threshold": threshold, "confusion_tn_fp_fn_tp": [tn, fp, fn, tp],
            "sensitivity": tp / (tp + fn), "specificity": tn / (tn + fp),
            "f1": 2 * tp / max(2 * tp + fp + fn, 1)}


def outer_folds(labels: pd.DataFrame, y: np.ndarray, group: str):
    indices = np.flatnonzero(np.array([group_of(r) == group for r in labels.region]))
    groups = labels.study_key.to_numpy()[indices]
    cv = StratifiedGroupKFold(5, shuffle=True, random_state=SEED)
    for fold, (train, test) in enumerate(cv.split(indices, y[indices], groups), 1):
        train_idx, test_idx = indices[train], indices[test]
        if set(labels.study_key.iloc[train_idx]) & set(labels.study_key.iloc[test_idx]):
            raise RuntimeError("Study group leaked across train and test")
        yield fold, train_idx, test_idx


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--labels", type=Path, default=Path("/app/labels/image_labels.csv"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--fit-final-only", action="store_true",
                        help="fit on all labelled images and save experimental heads; no validation")
    args = parser.parse_args()
    warnings.filterwarnings("ignore", message="Invalid value for VR UI")
    if args.epochs < 1 or args.threads < 1:
        parser.error("epochs and threads must be positive")
    torch.set_num_threads(args.threads)
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)
    labels = valid_labels(args.labels)
    source_sha = hashlib.sha256(args.labels.read_bytes()).hexdigest()
    started = time.monotonic()
    features, changed_hashes = frozen_features(args.dataset, labels)
    import torchvision
    pretrained = torchvision.models.resnet18()
    pretrained.load_state_dict(torch.load(WEIGHTS, map_location="cpu", weights_only=False))
    y = labels.quality_class.to_numpy(dtype=int)
    args.output.mkdir(parents=True, exist_ok=True)
    if args.fit_final_only:
        for group in ("spine", "hip"):
            indices = np.flatnonzero(np.array([group_of(r) == group for r in labels.region]))
            model = train_head(features, y, indices, pretrained.layer4, args.epochs, SEED)
            torch.save(model.state_dict(), args.output / f"cnn_{group}_experimental.pt")
        manifest = {"purpose": "experimental only; no deployment decision",
                    "architecture": "ResNet18 ImageNet stem through layer3 + fine-tuned layer4 and binary head",
                    "epochs": args.epochs, "size": SIZE, "labels_sha256": source_sha,
                    "encoder_sha256": WEIGHTS_SHA256, "current_pixel_hash_mismatches": changed_hashes,
                    "weights": ["cnn_spine_experimental.pt", "cnn_hip_experimental.pt"]}
        (args.output / "cnn_weights_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
        print(json.dumps(manifest, ensure_ascii=False, indent=2), flush=True)
        return
    scores = np.full(len(labels), np.nan)
    folds = np.full(len(labels), -1, dtype=int)
    for group in ("spine", "hip"):
        indices = np.flatnonzero(np.array([group_of(r) == group for r in labels.region]))
        for fold, train_idx, test_idx in outer_folds(labels, y, group):
            print(f"{group} fold {fold}/5: train {len(train_idx)}, test {len(test_idx)}", flush=True)
            model = train_head(features, y, train_idx, pretrained.layer4, args.epochs, SEED + fold)
            with torch.inference_mode():
                scores[test_idx] = torch.sigmoid(model(features[test_idx])).numpy()
            folds[test_idx] = fold
        print(f"{group}: {metrics(y[indices], scores[indices])}", flush=True)
    if not np.isfinite(scores).all() or (folds < 1).any():
        raise RuntimeError("Some images did not receive a held-out score")
    report = {"scope": "organiser DXA only; known anatomical region; image-only ResNet18 layer4 fine-tuning",
              "limitations": "study-group CV, patient independence unverified; five fixed epochs; no external clinical validation",
              "seed": SEED, "epochs": args.epochs, "input_size": SIZE, "threads": args.threads,
              "labels_sha256": source_sha, "encoder_sha256": WEIGHTS_SHA256,
              "current_pixel_hash_mismatches": changed_hashes,
              "split": "StratifiedGroupKFold(5, shuffle=True, random_state=17), separately by spine/hip",
              "overall": metrics(y, scores),
              "regions": {group: metrics(y[idx], scores[idx]) for group in ("spine", "hip")
                          for idx in [np.flatnonzero(np.array([group_of(r) == group for r in labels.region]))]},
              "elapsed_seconds": round(time.monotonic() - started, 1)}
    table = labels[["pixel_sha256", "study_key", "region", "quality_class", "first_source_path"]].copy()
    table["fold"] = folds
    table["cnn_score"] = scores
    table.to_csv(args.output / "cnn_oof.csv", index=False)
    (args.output / "cnn_metrics.json").write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
