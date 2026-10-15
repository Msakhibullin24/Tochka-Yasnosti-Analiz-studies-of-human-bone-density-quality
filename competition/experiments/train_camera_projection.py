"""Train endpoint AP/lateral recognition on pelvis-referenced synthetic views.

One fixed seed/protocol. Intermediate angles have no clinical target class and
are reported as response curves only. No DXA accuracy is established here.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score


class ProjectionNet(nn.Module):
    def __init__(self):
        super().__init__()
        layers = []
        channels = 2
        for width in (16, 32, 64):
            layers.extend([nn.Conv2d(channels, width, 3, padding=1), nn.GroupNorm(4, width),
                           nn.SiLU(), nn.AvgPool2d(2)])
            channels = width
        self.features = nn.Sequential(*layers)
        self.head = nn.Sequential(nn.AdaptiveAvgPool2d(4), nn.Flatten(), nn.Linear(1024, 64), nn.SiLU(), nn.Linear(64, 2))

    def forward(self, x):
        return self.head(self.features(x))


def augment(images):
    n = len(images)
    angle = (torch.rand(n, device=images.device) - .5) * np.pi / 9
    scale = .85 + .3 * torch.rand(n, device=images.device)
    flip = torch.where(torch.rand(n, device=images.device) > .5, 1., -1.)
    affine = torch.zeros(n, 2, 3, device=images.device)
    affine[:, 0, 0] = angle.cos() / scale * flip
    affine[:, 1, 0] = angle.sin() / scale * flip
    affine[:, 0, 1] = -angle.sin() / scale
    affine[:, 1, 1] = angle.cos() / scale
    affine[:, :, 2] = (torch.rand(n, 2, device=images.device) - .5) * .1
    images = F.grid_sample(images, F.affine_grid(affine, images.shape, align_corners=False),
                           mode='nearest', align_corners=False)
    # Partial loss of the pelvic mask is possible in actual proximal scans.
    images[:, 1] *= (torch.rand(n, device=images.device) > .3)[:, None, None]
    return images


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('Choose a new output directory')
    source = (args.dataset / 'dataset.json').read_bytes()
    metadata = json.loads(source)
    array_path = args.dataset / 'masks.npy'
    if hashlib.sha256(array_path.read_bytes()).hexdigest() != metadata['masks_sha256']:
        raise ValueError('Synthetic masks checksum differs')
    masks = np.load(array_path)[:, :2].copy()
    rows = metadata['records']
    train = [i for i, r in enumerate(rows) if r['split'] == 'train' and r['view'] != 'intermediate']
    held = [i for i, r in enumerate(rows) if r['split'] == 'test' and r['view'] != 'intermediate']
    if {rows[i]['subject'] for i in train} & {rows[i]['subject'] for i in held}:
        raise ValueError('Donor overlap')
    torch.manual_seed(17)
    np.random.seed(17)
    torch.set_num_threads(4)
    x = torch.tensor(masks[train], dtype=torch.float32, device='cuda')
    y = torch.tensor([rows[i]['view'] == 'lateral' for i in train], dtype=torch.long, device='cuda')
    model = ProjectionNet().cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.001)
    weights = len(train) / (2 * torch.bincount(y).float())
    started = perf_counter()
    for epoch in range(150):
        model.train()
        losses = []
        for ids in torch.randperm(len(train), device='cuda').split(32):
            loss = F.cross_entropy(model(augment(x[ids])), y[ids], weight=weights)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))
        if (epoch + 1) % 50 == 0:
            print(json.dumps({'epoch': epoch + 1, 'training_loss': float(np.mean(losses))}), flush=True)
    model.eval()
    probabilities = []
    with torch.inference_mode():
        for batch in np.array_split(masks, max(1, len(masks) // 32)):
            probabilities.extend(model(torch.tensor(batch, dtype=torch.float32, device='cuda')).softmax(1)[:, 1].cpu().tolist())
    probabilities = np.asarray(probabilities)
    truth = np.array([rows[i]['view'] == 'lateral' for i in held])
    score = probabilities[held]
    args.output.mkdir(parents=True)
    torch.save({'schema_version': 1, 'state_dict': {k: v.cpu() for k, v in model.state_dict().items()},
                'channels': ['femur', 'ipsilateral_pelvis'], 'classes': ['frontal_endpoint', 'lateral_endpoint'],
                'normalization': 'physical_proximal_mask150mm_10mm_pad', 'clinical_validation': False}, args.output / 'candidate.pt')
    report = {'protocol': __doc__, 'training_images': len(train), 'held_out_images': len(held),
              'training_donors': len({rows[i]['subject'] for i in train}),
              'held_out_donors': len({rows[i]['subject'] for i in held}),
              'accuracy': float(accuracy_score(truth, score >= .5)), 'f1_lateral': float(f1_score(truth, score >= .5)),
              'roc_auc': float(roc_auc_score(truth, score)),
              'held_donor_response_by_camera': {str(angle): {'n': int(sum(r['split'] == 'test' and r['camera_degrees'] == angle for r in rows)),
                    'mean_lateral_probability': float(np.mean([probabilities[i] for i, r in enumerate(rows) if r['split'] == 'test' and r['camera_degrees'] == angle]))}
                    for angle in sorted({r['camera_degrees'] for r in rows})},
              'model_sha256': hashlib.sha256((args.output / 'candidate.pt').read_bytes()).hexdigest(),
              'dataset_sha256': hashlib.sha256(source).hexdigest(),
              'code_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'seconds': perf_counter() - started, 'clinical_validation': False, 'automatic_promotions': 0,
              'limitations': metadata['limitations'],
              'held_predictions': [{**rows[i], 'lateral_probability': float(probabilities[i])} for i in held]}
    (args.output / 'evaluation.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({k: report[k] for k in ('accuracy', 'f1_lateral', 'roc_auc', 'seconds')}), flush=True)


if __name__ == '__main__':
    main()
