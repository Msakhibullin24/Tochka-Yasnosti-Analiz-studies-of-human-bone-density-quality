"""Train an unbiased spine-tilt regressor on synthetic rotations.

The diagnosis so far: spine_axis has ROC AUC 0.834 but the official >5 deg rule
catches 1 of 10 positives, because the hand-crafted angle estimator shrinks large
angles toward zero. Ten labelled positives are far too few to fit a replacement.

Synthetic data removes that constraint. Rotating a frame about its centre by a
known angle produces an image whose spine tilt is known exactly, so an unlimited
exactly-labelled training set can be built from the 249 real frames without ever
touching a real label. A model fitted on that set cannot inherit the hand-crafted
estimator's shrinkage, because its target is the applied rotation itself.

The trained regressor is then applied to the unrotated frames. That application is
the honest test:

  * if the regressor separates the 89 aligned from the 10 misaligned spines at
    all, the measurement defect is real and the criterion is recoverable;
  * if it does not, the labels must mean something other than visible tilt in the
    frame, and no amount of modelling will recover it.

Either answer is decisive, and neither uses a real label for fitting.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
from dxaqc.dicom_io import read_any  # noqa: E402
from source_integrity import inspect_sources  # noqa: E402

SIZE = 224


def letterbox(pixels, size=SIZE):
    import cv2
    h, w = pixels.shape
    scale = min(size / w, size / h)
    nw, nh = min(size, max(1, round(w * scale))), min(size, max(1, round(h * scale)))
    small = cv2.resize(pixels, (nw, nh), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((size, size), np.uint8)
    y, x = (size - nh) // 2, (size - nw) // 2
    canvas[y:y + nh, x:x + nw] = small
    return canvas


def rotate(image, degrees):
    import cv2
    matrix = cv2.getRotationMatrix2D((image.shape[1] / 2, image.shape[0] / 2), degrees, 1.0)
    return cv2.warpAffine(image, matrix, (image.shape[1], image.shape[0]),
                          flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)


class TiltNet(nn.Module):
    """Small residual CNN: enough capacity for orientation, small enough for 249 frames."""

    def __init__(self, width=32):
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
        self.head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(),
                                  nn.Linear(width * 8, 64), nn.ReLU(inplace=True),
                                  nn.Linear(64, 1))

    def forward(self, x):
        return self.head(self.body(self.stem(x))).squeeze(1)


def load_frames(dataset, labels, audited):
    frames = []
    for row in labels.itertuples():
        image = read_any(dataset / row.first_source_path)
        if image.pixel_sha256 != audited[row.first_source_path]:
            raise ValueError('Source pixels changed after audit')
        pixels = np.ascontiguousarray(image.pixels[:, ::-1] if row.region == 'hip_left' else image.pixels)
        frames.append(letterbox(pixels))
    return frames


def build_synthetic(frames, angles_per_frame, seed, only_spine):
    """Return rotated tensors and their exact signed rotation labels."""
    rng = np.random.default_rng(seed)
    images, targets = [], []
    for index, frame in enumerate(frames):
        if only_spine[index]:
            for angle in angles_per_frame:
                images.append(rotate(frame, float(angle)))
                targets.append(float(angle))
        else:
            # Non-spine frames are still valid negatives at zero rotation: they teach
            # the network that a hip frame has no spine axis to report.
            images.append(frame)
            targets.append(0.0)
    stack = np.stack(images).astype(np.float32) / 255.0
    return torch.from_numpy(stack[:, None]), torch.tensor(targets, dtype=torch.float32)


def run(args):
    from sklearn.metrics import roc_auc_score
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device == 'cpu' and args.epochs > 5:
        raise RuntimeError('Use --epochs 5 or fewer for a CPU smoke run')
    if args.output.exists():
        raise ValueError('Choose a new experiment directory')
    args.output.mkdir(parents=True)
    labels = pd.read_csv(args.labels)
    labels = labels[labels.quality_class.notna()].reset_index(drop=True)
    audit = inspect_sources([('organiser', args.labels, args.dataset)])
    audited = {r['path']: r['pixel_sha256_current'] for r in audit['entries']}
    frames = load_frames(args.dataset, labels, audited)
    is_spine = np.array([r == 'spine' for r in labels.region])
    print(f'{len(frames)} frames, {is_spine.sum()} spine', flush=True)

    rng = np.random.default_rng(args.seed)
    angles = rng.uniform(-args.max_angle, args.max_angle, args.angles_per_frame)
    angles = angles[np.abs(angles) > 1.0]
    train_x, train_y = build_synthetic(frames, angles, args.seed, is_spine)

    # Held-out source frames: rotate them at angles never seen in training.
    hold = rng.choice(len(frames), size=max(1, len(frames) // 10), replace=False)
    test_angles = np.array([0.0, 3.0, 5.5, 8.0, 12.0, -5.5, -9.0])
    test_x, test_y, test_src = [], [], []
    for i in hold:
        for angle in test_angles:
            test_x.append(rotate(frames[i], float(angle)))
            test_y.append(float(angle))
            test_src.append(i)
    test_x = torch.from_numpy(np.stack(test_x).astype(np.float32)[:, None] / 255.0)
    test_y = torch.tensor(test_y, dtype=torch.float32)

    model = TiltNet(args.width).to(device)
    optimiser = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=args.epochs)
    loss_fn = nn.SmoothL1Loss(beta=2.0)
    n = len(train_x)
    history = []
    for epoch in range(args.epochs):
        model.train()
        order = torch.randperm(n)
        losses = []
        for start in range(0, n, args.batch_size):
            batch = order[start:start + args.batch_size]
            xb = train_x[batch].to(device)
            yb = train_y[batch].to(device)
            optimiser.zero_grad(set_to_none=True)
            loss = loss_fn(model(xb), yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            optimiser.step()
            losses.append(float(loss.detach()))
        scheduler.step()
        model.eval()
        with torch.no_grad():
            predicted = model(test_x.to(device)).cpu().numpy()
        mae = float(np.abs(predicted - test_y.numpy()).mean())
        history.append({'epoch': epoch + 1, 'train_loss': float(np.mean(losses)),
                        'held_out_mae_deg': mae})
        print(f'epoch {epoch + 1}/{args.epochs} loss {np.mean(losses):.3f} held-out MAE {mae:.2f} deg',
              flush=True)

    # Apply to the untouched frames. This is the honest evaluation.
    model.eval()
    with torch.no_grad():
        estimate = model(torch.from_numpy(np.stack(frames).astype(np.float32)[:, None] / 255.0).to(device)).cpu().numpy()
    tilt = np.abs(estimate)
    truth = pd.to_numeric(labels.spine_axis, errors='coerce').to_numpy(float)
    mask = is_spine & np.isfinite(truth)
    positives = np.zeros(len(labels), bool)
    positives[mask] = truth[mask] == 1
    aligned = mask & ~positives
    result = {
        'protocol': __doc__,
        'architecture': {'width': args.width, 'epochs': args.epochs, 'batch_size': args.batch_size,
                         'max_angle': args.max_angle, 'angles_per_frame': args.angles_per_frame,
                         'synthetic_train_samples': int(n), 'device': device},
        'history': history, 'synthetic_held_out_mae_deg': history[-1]['held_out_mae_deg'],
        'real_frame_evaluation': {
            'spine_rows': int(mask.sum()), 'positives': int(positives.sum()),
            'roc_auc_of_regressor_on_spine_axis': float(roc_auc_score(positives[mask].astype(int), tilt[mask])),
            'predicted_abs_tilt_deg': {
                'aligned_median': float(np.median(tilt[aligned])),
                'aligned_p90': float(np.percentile(tilt[aligned], 90)),
                'misaligned_median': float(np.median(tilt[mask & positives])) if positives.sum() else None,
                'misaligned_min': float(tilt[mask & positives].min()) if positives.sum() else None,
                'misaligned_max': float(tilt[mask & positives].max()) if positives.sum() else None},
            'positives_over_5deg': int((tilt[mask & positives] > 5).sum()),
            'negatives_over_5deg': int((tilt[aligned] > 5).sum()),
            'best_threshold_positives_caught': None}}
    if positives.sum():
        best, chosen = -1.0, None
        for threshold in np.unique(tilt[mask]):
            caught = int(((tilt[mask] >= threshold) & positives[mask]).sum())
            false_alarm = int(((tilt[mask] >= threshold) & ~positives[mask]).sum())
            f1 = 2 * caught / max(2 * caught + (int(positives.sum()) - caught) + false_alarm, 1)
            if f1 > best:
                best, chosen = f1, float(threshold)
        result['real_frame_evaluation']['best_threshold_positives_caught'] = {
            'threshold_deg': chosen, 'f1_on_this_cohort': best,
            'note': 'Selected on the same 99 rows it is measured on, so this is an '
                    'upper bound on what the measurement could deliver, not a validated score.'}
    np.save(args.output / 'tilt_estimates.npy', estimate.astype(np.float32))
    pd.DataFrame({'path': labels.first_source_path, 'region': labels.region,
                  'predicted_signed_tilt_deg': estimate,
                  'predicted_abs_tilt_deg': tilt,
                  'spine_axis_label': truth}).to_csv(args.output / 'tilt_estimates.csv', index=False)
    torch.save({'state_dict': model.state_dict(), 'width': args.width,
                'clinical_validation': False}, args.output / 'tiltnet.pt')
    (args.output / 'report.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(result['real_frame_evaluation'], indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--labels', type=Path, default=HERE / 'labels/image_labels.csv')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--width', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=40)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--angles-per-frame', type=int, default=120)
    parser.add_argument('--max-angle', type=float, default=30.)
    parser.add_argument('--seed', type=int, default=17)
    run(parser.parse_args())