"""Fixed donor-exclusive landmark ablations on APP camera/leg projections.

Only camera angles -30,0,30 degrees are used. Compare old femur-only model on
the exact same held views. No new independent donors or clinical DXA truth.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from time import perf_counter

import numpy as np
import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.train_silhouette_heatmaps import Heatmaps, ORDER, augment, coordinates, predict


def predict_channels(model, masks, device='cuda'):
    with torch.inference_mode():
        points, _ = coordinates(model(torch.tensor(np.ascontiguousarray(masks), dtype=torch.float32, device=device)))
    return (points.cpu().numpy() + 1) * 64 - .5


def errors(prediction, truth, scales):
    distance = np.linalg.norm(prediction - truth, axis=2) * scales[:, None]
    p, t = prediction[:, 2] - prediction[:, 3], truth[:, 2] - truth[:, 3]
    denom = np.linalg.norm(p, axis=1) * np.linalg.norm(t, axis=1)
    valid = (np.linalg.norm(t, axis=1) * scales >= 5) & (np.linalg.norm(p, axis=1) * scales >= 1)
    cosine = np.divide((p * t).sum(1), denom, out=np.zeros(len(p)), where=denom > 0)
    angle = np.degrees(np.arccos(np.clip(cosine[valid], -1, 1)))
    return {'mean_point_error_mm': dict(zip(ORDER, distance.mean(0).tolist())),
            'mean_neck_axis_error_degrees': float(angle.mean()) if len(angle) else None,
            'p95_neck_axis_error_degrees': float(np.percentile(angle, 95)) if len(angle) else None,
            'axis_geometrically_identifiable_cases': int(valid.sum()), 'cases': len(prediction)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('dataset', 'baseline_model', 'output'):
        parser.add_argument('--' + name.replace('_', '-'), type=Path, required=True)
    parser.add_argument('--channels', type=int, choices=(1, 2), required=True)
    parser.add_argument('--mask-noise', action='store_true')
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('Choose a new output directory')
    metadata_path = args.dataset / 'dataset.json'
    metadata = json.loads(metadata_path.read_text())
    if hashlib.sha256((args.dataset / 'masks.npy').read_bytes()).hexdigest() != metadata['masks_sha256']:
        raise ValueError('Synthetic masks checksum differs')
    records = metadata['records']
    selection = [i for i, r in enumerate(records) if abs(r['camera_degrees']) <= 30]
    records = [records[i] for i in selection]
    masks = np.load(args.dataset / 'masks.npy')[selection, :args.channels].copy()
    train = np.array([i for i, r in enumerate(records) if r['split'] == 'train'])
    held = np.array([i for i, r in enumerate(records) if r['split'] == 'test'])
    if {records[i]['subject'] for i in train} & {records[i]['subject'] for i in held}:
        raise ValueError('Donors overlap')
    points = np.array([r['landmark_points_128'] for r in records])
    if not np.isfinite(points).all() or np.any(points < -.5) or np.any(points > 127.5):
        raise ValueError('Synthetic landmark falls outside the source canvas')
    scales = np.array([r['normalized_pixel_mm'] for r in records])
    torch.set_num_threads(4)
    torch.manual_seed(17)
    np.random.seed(17)
    x = torch.tensor(masks[train], dtype=torch.float32, device='cuda')
    y = torch.tensor((points[train] + .5) / 64 - 1, dtype=torch.float32, device='cuda')
    model = Heatmaps(args.channels).cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=.002, weight_decay=.0001)
    started = perf_counter()
    for epoch in range(300):
        model.train()
        losses = []
        for ids in torch.randperm(len(train), device='cuda').split(16):
            images, truth = augment(x[ids], y[ids])
            invalid = (truth.abs() > .98).any(dim=(1, 2))
            images[invalid], truth[invalid] = x[ids][invalid], y[ids][invalid]
            if args.channels == 2:
                images[:, 1] *= (torch.rand(len(ids), device='cuda') > .3)[:, None, None]
            if args.mask_noise:
                eroded = 1 - F.max_pool2d(1 - images[:, :1], 3, stride=1, padding=1)
                dilated = F.max_pool2d(images[:, :1], 3, stride=1, padding=1)
                choices = torch.randint(3, (len(ids), 1, 1, 1), device='cuda')
                images[:, :1] = torch.where(choices == 0, eroded, torch.where(choices == 1, dilated, images[:, :1]))
            logits = model(images)
            predicted, grid = coordinates(logits)
            target = (-((grid[None, None] - truth[:, :, None])**2).sum(-1) / (2 * .025**2)).softmax(-1)
            loss = -(target * logits.flatten(2).log_softmax(-1)).sum(-1).mean() + 5 * F.mse_loss(predicted, truth)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))
        if (epoch + 1) % 50 == 0:
            print(json.dumps({'epoch': epoch + 1, 'training_loss': float(np.mean(losses))}), flush=True)
    model.eval()
    prediction = np.concatenate([predict_channels(model, masks[ids]) for ids in np.array_split(held, max(1, len(held) // 8))])
    baseline_checkpoint = torch.load(args.baseline_model, map_location='cpu', weights_only=True)
    baseline = Heatmaps().cuda().eval()
    baseline.load_state_dict(baseline_checkpoint['state_dict'], strict=True)
    baseline_prediction = np.concatenate([predict(baseline, masks[ids, 0]) for ids in np.array_split(held, max(1, len(held) // 8))])
    args.output.mkdir(parents=True)
    torch.save({'schema_version': 1, 'state_dict': {k: v.cpu() for k, v in model.state_dict().items()},
                'input_channels': args.channels, 'landmark_order': list(ORDER),
                'normalization': 'physical_proximal_mask150mm_10mm_pad', 'clinical_validation': False,
                'domain': 'synthetic APP camera +-30deg and mechanical-axis leg rotation +-30deg',
                'channels': ['femur', 'ipsilateral_pelvis'][:args.channels]}, args.output / 'candidate.pt')
    result = {'protocol': __doc__, 'training_images': len(train), 'held_out_images': len(held),
              'training_donors': len({records[i]['subject'] for i in train}), 'held_out_donors': len({records[i]['subject'] for i in held}),
              'input_channels': args.channels, 'mask_noise': args.mask_noise,
              'candidate': errors(prediction, points[held], scales[held]),
              'paired_baseline': errors(baseline_prediction, points[held], scales[held]),
              'model_sha256': hashlib.sha256((args.output / 'candidate.pt').read_bytes()).hexdigest(),
              'baseline_sha256': hashlib.sha256(args.baseline_model.read_bytes()).hexdigest(),
              'dataset_sha256': hashlib.sha256(metadata_path.read_bytes()).hexdigest(),
              'code_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'architecture_code_sha256': hashlib.sha256(Path(__file__).with_name('train_silhouette_heatmaps.py').read_bytes()).hexdigest(),
              'seconds': perf_counter() - started, 'clinical_validation': False, 'automatic_promotions': 0,
              'limitations': metadata['limitations'] + ['Same four previously inspected held-out donors; not independent clinical data.']}
    (args.output / 'evaluation.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
