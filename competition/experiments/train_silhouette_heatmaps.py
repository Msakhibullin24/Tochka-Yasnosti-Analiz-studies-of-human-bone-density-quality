"""Train a small heatmap landmark network on published 3D-derived silhouettes.

Fixed protocol: 300 epochs, seed17, existing 15/4 subject split. No selection
on held-out landmark error; four held-out subjects have been inspected before.
This is a synthetic-geometry experiment, not DXA landmark validation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from time import perf_counter

import cv2
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


ORDER = ('greater_trochanter', 'lesser_trochanter', 'head_center', 'neck_isthmus_center')


def block(inputs, outputs):
    return nn.Sequential(nn.Conv2d(inputs, outputs, 3, padding=1), nn.GroupNorm(4, outputs), nn.SiLU(),
                         nn.Conv2d(outputs, outputs, 3, padding=1), nn.GroupNorm(4, outputs), nn.SiLU())


class Heatmaps(nn.Module):
    def __init__(self, input_channels=1):
        super().__init__()
        self.input_channels = input_channels
        self.a, self.b, self.c = block(input_channels, 16), block(16, 32), block(32, 64)
        self.up = block(96, 32)
        self.out = nn.Conv2d(32, 4, 1)

    def forward(self, x):
        a = self.a(x)
        b = self.b(F.avg_pool2d(a, 2))
        c = self.c(F.avg_pool2d(b, 2))
        return self.out(self.up(torch.cat([F.interpolate(c, size=b.shape[-2:], mode='bilinear', align_corners=False), b], 1)))


def coordinates(logits):
    height, width = logits.shape[-2:]
    y, x = torch.meshgrid((torch.arange(height, device=logits.device) + .5) / height * 2 - 1,
                          (torch.arange(width, device=logits.device) + .5) / width * 2 - 1, indexing='ij')
    grid = torch.stack([x, y], -1).reshape(-1, 2)
    probabilities = logits.flatten(2).softmax(-1)
    return probabilities @ grid, grid


def augment(images, points):
    n = len(images)
    angle = (torch.rand(n, device=images.device) - .5) * np.pi / 6
    scale = .85 + torch.rand(n, device=images.device) * .3
    flip = torch.where(torch.rand(n, device=images.device) > .5, 1., -1.)
    forward = torch.eye(3, device=images.device)[None].repeat(n, 1, 1)
    forward[:, 0, 0] = angle.cos() * scale * flip
    forward[:, 0, 1] = -angle.sin() * scale
    forward[:, 1, 0] = angle.sin() * scale * flip
    forward[:, 1, 1] = angle.cos() * scale
    forward[:, :2, 2] = (torch.rand(n, 2, device=images.device) - .5) * .15
    inverse = torch.linalg.inv(forward)[:, :2]
    grid = F.affine_grid(inverse, images.shape, align_corners=False)
    images = F.grid_sample(images, grid, align_corners=False)
    points = torch.einsum('nij,nkj->nki', forward[:, :2, :2], points) + forward[:, None, :2, 2]
    return images, points


def predict(model, masks, device='cuda'):
    masks = np.asarray(masks)
    if masks.ndim != 3:
        raise ValueError('Expected a batch of grayscale masks')
    x = torch.from_numpy(np.stack([cv2.resize((m > 0).astype(np.float32), (128, 128)) for m in masks]))[:, None].to(device)
    with torch.inference_mode():
        points, _ = coordinates(model(x))
    return ((points.cpu().numpy() + 1) / 2 * np.array([masks.shape[2], masks.shape[1]]) - .5)


def normalize_proximal_mask(mask, sx, sy):
    """Map proximal 150 mm of a mask to a square while preserving physical shape.

    Returns an explicit original-pixel -> normalized-pixel affine transform.
    Ten mm of canvas padding is geometric context, not measured source tissue.
    """
    if not all(np.isfinite(v) and v > 0 for v in (sx, sy)):
        raise ValueError('Known positive pixel calibration required')
    mask = np.asarray(mask, dtype=np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    if n <= 1:
        raise ValueError('Nonempty femur mask required')
    mask = labels == 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    ys, xs = np.where(mask)
    y0 = int(ys.min())
    y1 = min(mask.shape[0], y0 + int(np.ceil(150 / sy)))
    mask[y1:] = False
    ys, xs = np.where(mask)
    width, height = (xs.max() - xs.min() + 1) * sx, (ys.max() - ys.min() + 1) * sy
    if min(width, height) < 5:
        raise ValueError('Degenerate femur mask')
    scale = 320 / (max(width, height) + 20)
    center = np.array([(xs.min() + xs.max()) / 2, (ys.min() + ys.max()) / 2])
    affine = np.array([[scale * sx, 0., 159.5 - scale * sx * center[0]],
                       [0., scale * sy, 159.5 - scale * sy * center[1]]])
    image = cv2.warpAffine(mask.astype(np.uint8), affine, (320, 320), flags=cv2.INTER_NEAREST)
    return image, affine


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--normalize-proximal', action='store_true')
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('Choose a new output directory')
    torch.set_num_threads(4)
    torch.manual_seed(17)
    np.random.seed(17)
    metadata = json.loads((args.dataset / 'dataset.json').read_text())
    rows = metadata['records']
    train = np.array([i for i, row in enumerate(rows) if row['split'] == 'train'])
    test = np.array([i for i, row in enumerate(rows) if row['split'] == 'test'])
    if not len(train) or not len(test) or {rows[i]['subject'] for i in train} & {rows[i]['subject'] for i in test}:
        raise ValueError('Subject split missing or overlapping')
    masks, points, original_points, inverse_transforms = [], [], [], []
    for row in rows:
        path = (args.dataset / row['image']).resolve()
        if not path.is_relative_to(args.dataset.resolve()):
            raise ValueError('Image path outside dataset')
        mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if mask is None or mask.shape != (320, 320):
            raise ValueError('Expected published-projection 320x320 raster')
        point = np.vstack([row['landmarks']['greater_trochanter'], row['landmarks']['lesser_trochanter'],
                           row['landmarks']['femoral_neck_axis']])
        original_points.append(point.copy())
        affine = np.array([[1., 0., 0.], [0., 1., 0.]])
        if args.normalize_proximal:
            mask, affine = normalize_proximal_mask(mask, row['pixel_mm_x'], row['pixel_mm_y'])
            point = point @ affine[:, :2].T + affine[:, 2]
        inverse_transforms.append(cv2.invertAffineTransform(affine))
        masks.append(mask)
        points.append(point)
    # Only training targets are materialized on the GPU.
    x = torch.from_numpy(np.stack([cv2.resize((masks[i] > 0).astype(np.float32), (128, 128)) for i in train]))[:, None].cuda()
    target = torch.tensor((np.asarray(points)[train] + .5) / 320 * 2 - 1, dtype=torch.float32, device='cuda')
    model = Heatmaps().cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=.002, weight_decay=.0001)
    started = perf_counter()
    for epoch in range(300):
        model.train()
        losses = []
        for selection in torch.randperm(len(train), device='cuda').split(16):
            images, truth = augment(x[selection], target[selection])
            logits = model(images)
            predicted, grid = coordinates(logits)
            gaussian = (-((grid[None, None] - truth[:, :, None])**2).sum(-1) / (2 * .025**2)).softmax(-1)
            loss = -(gaussian * logits.flatten(2).log_softmax(-1)).sum(-1).mean() + 5 * F.mse_loss(predicted, truth)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))
        if (epoch + 1) % 50 == 0:
            print(json.dumps({'epoch': epoch + 1, 'train_loss': float(np.mean(losses)), 'seconds': perf_counter() - started}), flush=True)
    model.eval()
    prediction = np.concatenate([predict(model, np.asarray(masks)[ids]) for ids in np.array_split(test, max(1, len(test) // 8))])
    for j, index in enumerate(test):
        inverse = inverse_transforms[index]
        prediction[j] = prediction[j] @ inverse[:, :2].T + inverse[:, 2]
    truth = np.asarray(original_points)[test]
    error = np.linalg.norm((prediction - truth) * .5, axis=2)
    pa, ta = prediction[:, 2] - prediction[:, 3], truth[:, 2] - truth[:, 3]
    denominator = np.linalg.norm(pa, axis=1) * np.linalg.norm(ta, axis=1)
    cosine = np.divide((pa * ta).sum(1), denominator, out=np.zeros(len(pa)), where=denominator > 0)
    angles = np.degrees(np.arccos(np.clip(cosine, -1, 1)))
    args.output.mkdir(parents=True)
    checkpoint = {'schema_version': 1, 'state_dict': {k: v.cpu() for k, v in model.state_dict().items()},
                  'input_size': 128, 'landmark_order': list(ORDER), 'clinical_validation': False,
                  'normalization': 'physical_proximal_mask150mm_10mm_pad' if args.normalize_proximal else 'raw_320_raster',
                  'domain': 'synthetic orthographic proximal femur silhouettes'}
    torch.save(checkpoint, args.output / 'candidate.pt')
    report = {'protocol': __doc__, 'training_images': len(train), 'held_out_images': len(test),
              'training_subjects': len({rows[i]['subject'] for i in train}), 'test_subjects': len({rows[i]['subject'] for i in test}),
              'mean_point_error_mm': dict(zip(ORDER, error.mean(0).tolist())),
              'mean_neck_axis_error_degrees': float(angles.mean()), 'p95_neck_axis_error_degrees': float(np.percentile(angles, 95)),
              'model_sha256': hashlib.sha256((args.output / 'candidate.pt').read_bytes()).hexdigest(),
              'dataset_sha256': hashlib.sha256((args.dataset / 'dataset.json').read_bytes()).hexdigest(),
              'code_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'seconds': perf_counter() - started, 'clinical_validation': False, 'automatic_promotions': 0,
              'normalization': checkpoint['normalization'],
              'test_predictions': [{'subject': rows[i]['subject'], 'image': rows[i]['image'],
                                    'points': prediction[j].tolist(), 'truth': truth[j].tolist()} for j, i in enumerate(test)],
              'limitations': metadata['limitations'] + ['Previously inspected held-out subjects; no independent new cohort.',
                                                       'No DXA segmentation errors or pelvic superposition in this benchmark.']}
    (args.output / 'evaluation.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({k: report[k] for k in ('mean_point_error_mm', 'mean_neck_axis_error_degrees', 'seconds')}), flush=True)


if __name__ == '__main__':
    main()
