"""E14 synthetic artifact supervision for spine_artifact.

Why this criterion and not coverage. spine_artifact is the one criterion where every
learned candidate in this project has been *worse* than the baseline: 0.7568 down to
0.6823 averaged over seeds. It has 17 positives and the best of 15 oracle photometric
statistics reaches only AUC 0.667, so hand-crafted measurements barely separate it. By
contrast the coverage criteria already separate at AUC 0.84 and 0.94 and fail on the
threshold, which more training data cannot fix. Artifacts are therefore where
synthetic supervision has a mechanism to work with.

The pretext is not a stand-in for the label. Real metal produces compact bright
structures with radial streaking and bloom; straps and clothing produce dark occlusion;
vendor ROI overlays produce thin bright lines that are *not* defects and are the
critical hard negative. All three are generated explicitly, with the label known by
construction, and every variant of a frame stays inside that frame's study so no child
can cross a fold boundary.

Acceptance is honest: the detector is validated on held-out synthetic frames first,
then its score on the 249 real frames is reported against the real spine_artifact
labels. If the score does not separate the real labels, the domain gap has failed and
the score will not be transferred into the screen.
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
from torch import nn

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
from dxaqc.dicom_io import read_any  # noqa: E402
from gpu_research.common import digest, dump  # noqa: E402
from source_integrity import inspect_sources  # noqa: E402

SIZE = 192
KINDS = ('metal', 'occlusion', 'roi_line', 'clean')


def ellipse_patch(rng, height, width, angle, softness):
    """A rotated elliptical intensity field, used as the base of every object."""
    ys, xs = np.mgrid[0:height, 0:width].astype(np.float32)
    cy, cx = height / 2, width / 2
    dy, dx = ys - cy, xs - cx
    ca, sa = np.cos(angle), np.sin(angle)
    u = dx * ca - dy * sa
    v = dx * sa + dy * ca
    field = 1.0 - (u / (width / 2)) ** 2 - (v / (height / 2)) ** 2
    return np.clip(field, 0, 1) ** max(softness, 0.05)


def make_metal(rng, pixels):
    """Metal artefact: compact bright body, radial streaks, and bloom."""
    h, w = pixels.shape
    out = pixels.astype(np.float32).copy()
    r = int(rng.integers(6, 16))
    patch = ellipse_patch(rng, 2 * r, 2 * r, rng.uniform(0, np.pi), rng.uniform(0.4, 1.2))
    cy = int(rng.integers(r, h - r))
    cx = int(rng.integers(r, w - r))
    y0, x0 = cy - r, cx - r
    region = out[y0:y0 + 2 * r, x0:x0 + 2 * r]
    region += patch * rng.uniform(120, 220)
    out[y0:y0 + 2 * r, x0:x0 + 2 * r] = region
    # Radial streaks, the signature of a metal object in DXA.
    angle = rng.uniform(0, np.pi)
    length = int(rng.integers(r, 3 * r + 8))
    for offset in np.linspace(-1, 1, 5):
        ys = np.arange(h)
        xs = np.arange(w)
        line_y = cy + offset * rng.uniform(2, 6)
        slope = np.tan(angle + rng.uniform(-0.3, 0.3))
        distance = np.abs(xs[None, :] - cx - slope * (ys[:, None] - line_y))
        weight = np.exp(-(distance / rng.uniform(1.0, 3.0)) ** 2) * np.exp(-(ys[:, None] - line_y) ** 2 / (2 * length ** 2))
        out += weight * rng.uniform(25, 70)
    # Bloom.
    from scipy.ndimage import gaussian_filter
    out += gaussian_filter(out - pixels.astype(np.float32), rng.uniform(3, 8)) * rng.uniform(0.4, 1.2)
    return np.clip(out, 0, 255), True


def make_occlusion(rng, pixels):
    """Dark band: clothing strap, blanket edge, body part outside the scan zone."""
    h, w = pixels.shape
    out = pixels.astype(np.float32).copy()
    thickness = int(rng.integers(5, 28))
    horizontal = rng.random() < 0.5
    value = rng.uniform(0, 45)
    if horizontal:
        y = int(rng.integers(0, h - thickness))
        out[y:y + thickness, :] = np.minimum(out[y:y + thickness, :], value)
    else:
        x = int(rng.integers(0, w - thickness))
        out[:, x:x + thickness] = np.minimum(out[:, x:x + thickness], value)
    return out, True


def make_roi_line(rng, pixels):
    """Thin bright vendor overlay line: a hard negative, explicitly not a defect."""
    h, w = pixels.shape
    out = pixels.astype(np.float32).copy()
    thickness = int(rng.integers(1, 3))
    if rng.random() < 0.5:
        x = int(rng.integers(0, w - thickness))
        out[:, x:x + thickness] = np.maximum(out[:, x:x + thickness], rng.uniform(120, 200))
    else:
        y = int(rng.integers(0, h - thickness))
        out[y:y + thickness, :] = np.maximum(out[y:y + thickness, :], rng.uniform(120, 200))
    return out, True


def make_clean(rng, pixels):
    """Unmodified, or with mild noise and gain changes only."""
    out = pixels.astype(np.float32) * rng.uniform(0.9, 1.1) + rng.uniform(-4, 4)
    return np.clip(out, 0, 255), False


GENERATORS = {'metal': make_metal, 'occlusion': make_occlusion, 'roi_line': make_roi_line, 'clean': make_clean}
# ROI overlay lines are hard negatives: present in the image, absent from the label.
LABELLED_AS_DEFECT = {'metal': 1, 'occlusion': 1, 'roi_line': 0, 'clean': 0}


class ArtifactNet(nn.Module):
    def __init__(self, width=32, embedding=128):
        super().__init__()
        def block(i, o, stride=2):
            return nn.Sequential(nn.Conv2d(i, o, 3, stride, 1, bias=False),
                                 nn.BatchNorm2d(o), nn.ReLU(inplace=True),
                                 nn.Conv2d(o, o, 3, 1, 1, bias=False), nn.BatchNorm2d(o),
                                 nn.ReLU(inplace=True))
        self.stem = nn.Sequential(nn.Conv2d(1, width, 5, 2, 2, bias=False),
                                  nn.BatchNorm2d(width), nn.ReLU(inplace=True))
        self.body = nn.Sequential(block(width, width * 2), block(width * 2, width * 4),
                                  block(width * 4, width * 8), block(width * 8, width * 8))
        self.embedding = nn.Sequential(nn.Linear(width * 8, embedding), nn.ReLU(inplace=True),
                                      nn.Linear(embedding, embedding))
        self.head = nn.Linear(embedding, 1)
        self.feature_dim = width * 8

    def features(self, x):
        return torch.flatten(nn.functional.adaptive_avg_pool2d(self.body(self.stem(x)), 1), 1)

    def embed(self, x):
        return self.embedding(self.features(x))

    def forward(self, x):
        return self.head(self.embed(x)).squeeze(1)


def load_frames(dataset, labels, audited, size=SIZE):
    import cv2
    frames = []
    for row in labels.itertuples():
        image = read_any(dataset / row.first_source_path)
        if image.pixel_sha256 != audited[row.first_source_path]:
            raise ValueError('Source pixels changed after audit')
        pixels = np.ascontiguousarray(image.pixels[:, ::-1] if row.region == 'hip_left' else image.pixels)
        h, w = pixels.shape
        scale = size / max(h, w)
        resized = cv2.resize(pixels, (max(1, round(w * scale)), max(1, round(h * scale))),
                             interpolation=cv2.INTER_AREA)
        canvas = np.zeros((size, size), np.uint8)
        y, x = (size - resized.shape[0]) // 2, (size - resized.shape[1]) // 2
        canvas[y:y + resized.shape[0], x:x + resized.shape[1]] = resized
        frames.append(canvas)
    return frames


def resize_batch(array, size=SIZE):
    import cv2
    out = []
    for frame in array:
        h, w = frame.shape
        scale = size / max(h, w)
        resized = cv2.resize(frame, (max(1, round(w * scale)), max(1, round(h * scale))),
                             interpolation=cv2.INTER_AREA)
        canvas = np.zeros((size, size), np.float32)
        y, x = (size - resized.shape[0]) // 2, (size - resized.shape[1]) // 2
        canvas[y:y + resized.shape[0], x:x + resized.shape[1]] = resized
        out.append(canvas)
    return np.stack(out)


def generate(frames, per_frame, seed):
    """Synthetic artifact corpus. Returns images, defect labels, kind ids, parent index."""
    rng = np.random.default_rng(seed)
    images, labels, kinds, parents = [], [], [], []
    # Half the samples are unaltered or near-unaltered negatives so the model cannot
    # learn "busy texture" as a proxy for "artefact".
    for i, frame in enumerate(frames):
        for _ in range(per_frame):
            kind = KINDS[rng.integers(0, len(KINDS))]
            altered, _ = GENERATORS[kind](rng, frame)
            images.append(altered)
            labels.append(LABELLED_AS_DEFECT[kind])
            kinds.append(KINDS.index(kind))
            parents.append(i)
    return (np.stack(images).astype(np.float32) / 255.0,
            np.asarray(labels, np.float32), np.asarray(kinds, np.int64), np.asarray(parents, np.int64))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--labels', type=Path, default=HERE / 'labels/image_labels.csv')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--per-frame', type=int, default=24)
    parser.add_argument('--epochs', type=int, default=40)
    parser.add_argument('--batch-size', type=int, default=96)
    parser.add_argument('--width', type=int, default=32)
    parser.add_argument('--seed', type=int, default=17)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('Choose a new experiment directory')
    args.output.mkdir(parents=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    labels = pd.read_csv(args.labels)
    labels = labels[labels.quality_class.notna()].reset_index(drop=True)
    audit = inspect_sources([('organiser', args.labels, args.dataset)])
    audited = {r['path']: r['pixel_sha256_current'] for r in audit['entries']}
    fingerprint = hashlib.sha256(json.dumps(
        [(r['path'], r['pixel_sha256_current']) for r in audit['entries']],
        ensure_ascii=False).encode()).hexdigest()
    frames = load_frames(args.dataset, labels, audited)
    print(f'{len(frames)} frames', flush=True)

    images, y, kinds, parents = generate(frames, args.per_frame, args.seed)
    print(f'synthetic corpus: {images.shape}, defect rate {y.mean():.3f}', flush=True)

    # Split by parent frame so no child crosses the boundary.
    rng = np.random.default_rng(args.seed)
    unique = np.unique(parents)
    rng.shuffle(unique)
    hold = set(unique[:max(1, len(unique) // 10)].tolist())
    val_mask = np.array([p in hold for p in parents])
    train_idx = np.flatnonzero(~val_mask)
    val_idx = np.flatnonzero(val_mask)
    if set(parents[train_idx]) & set(parents[val_idx]):
        raise ValueError('Synthetic child crossed a split boundary')

    mean = images[train_idx].mean()
    std = images[train_idx].std() + 1e-6
    x = torch.from_numpy((images - mean) / std).unsqueeze(1)
    target = torch.from_numpy(y)
    model = ArtifactNet(args.width).to(device)
    optimiser = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=args.epochs)
    weight = torch.tensor([float((target[train_idx] == 0).sum()) / max(float((target[train_idx] == 1).sum()), 1.0)],
                          device=device)
    history = []
    for epoch in range(args.epochs):
        model.train()
        order = rng.permutation(train_idx)
        losses = []
        for start in range(0, len(order), args.batch_size):
            batch = order[start:start + args.batch_size]
            optimiser.zero_grad(set_to_none=True)
            logits = model(x[batch].to(device))
            loss = nn.functional.binary_cross_entropy_with_logits(
                logits.float(), target[batch].to(device), pos_weight=weight)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            optimiser.step()
            losses.append(float(loss.detach()))
        scheduler.step()
        model.eval()
        with torch.no_grad():
            scores = torch.sigmoid(model(x[val_idx].to(device))).cpu().numpy()
        truth = y[val_idx]
        from sklearn.metrics import average_precision_score, roc_auc_score
        history.append({'epoch': epoch + 1, 'train_loss': float(np.mean(losses)),
                        'val_ap': float(average_precision_score(truth, scores)) if truth.sum() else None,
                        'val_auc': float(roc_auc_score(truth, scores)) if len(np.unique(truth)) > 1 else None})
        if epoch % 5 == 0 or epoch == args.epochs - 1:
            h = history[-1]
            print(f"epoch {epoch + 1}/{args.epochs} loss {np.mean(losses):.4f} "
                  f"val AP {h['val_ap']:.3f} AUC {h['val_auc']:.3f}", flush=True)

    # Apply to the unaltered real frames. This is the honest test of transfer.
    real = torch.from_numpy((np.stack(frames).astype(np.float32) / 255.0 - mean) / std).unsqueeze(1)
    model.eval()
    with torch.no_grad():
        real_score = torch.sigmoid(model(real.to(device))).cpu().numpy()
        real_features = model.features(real.to(device)).cpu().numpy()
    truth = pd.to_numeric(labels.spine_artifact, errors='coerce').to_numpy(float)
    spine = labels.region.eq('spine').to_numpy()
    mask = spine & np.isfinite(truth)
    from sklearn.metrics import roc_auc_score
    transfer = {'spine_rows': int(mask.sum()), 'positives': int((truth[mask] == 1).sum()),
                'auc_on_real_spine_artifact': float(roc_auc_score((truth[mask] == 1).astype(int),
                                                                  real_score[mask])) if mask.sum() > 10 else None}
    transfer['passes_acceptance'] = bool(transfer['auc_on_real_spine_artifact'] and
                                         transfer['auc_on_real_spine_artifact'] > 0.70)
    print(json.dumps(transfer, indent=2), flush=True)

    path = args.output / 'features.npy'
    np.save(path, real_features.astype(np.float32), allow_pickle=False)
    dump(path.with_suffix('.json'), {
        'schema_version': 1, 'model': 'synthetic-artifact-detector',
        'repo': 'trained from scratch in this repository', 'revision': f'seed{args.seed}',
        'labels_sha256': digest(args.labels), 'source_fingerprint': fingerprint,
        'paths': labels.first_source_path.astype(str).tolist(),
        'features_sha256': digest(path), 'features_mean': float(mean), 'features_std': float(std),
        'features_size': SIZE, 'encoder_finetuned': False, 'dense_features': False,
        'smoke_only': False, 'pooling': 'global-average of the final block',
        'labels': ['synthetic-artifact-head-probability'],
        'artifact_probability': real_score.tolist(),
        'artifact_label_kinds': list(KINDS),
        'synthetic_per_frame': args.per_frame, 'epochs': args.epochs,
        'acceptance_on_real_labels': transfer,
        'clinical_validation': False, 'model_affects_decision': False})
    torch.save({'state_dict': model.state_dict(), 'width': args.width, 'mean': float(mean),
                'std': float(std), 'size': SIZE, 'feature_dim': model.feature_dim,
                'clinical_validation': False}, args.output / 'artifactnet.pt')
    dump(args.output / 'report.json', {
        'protocol': __doc__, 'synthetic_images': int(images.shape[0]),
        'synthetic_defect_rate': float(y.mean()),
        'train_images': int(len(train_idx)), 'val_images': int(len(val_idx)),
        'split': 'by parent frame, children never cross the boundary',
        'history': history, 'transfer_to_real': transfer,
        'verdict': ('If transfer.passes_acceptance is false the domain gap between generated and real artefacts is too '
                    'large and the feature will not be used in any screen.')})
    print('written', path)


if __name__ == '__main__':
    main()