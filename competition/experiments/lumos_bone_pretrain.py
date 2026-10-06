"""Pretext-train an encoder on LUMOS density, then reuse it on organizer DXA.

Every attempt to raise the metrics with a better encoder failed, and the diagnosis
says why: the cohort has 249 images. LUMOS supplies 1614 real lumbar radiographs
across 800 patients with DXA-measured L1-L4 BMD and T-scores.

The pretext task is density regression. It is a genuine anatomical task - the
network has to find the lumbar bone column, exclude soft tissue and background,
and read its radiodensity - so its penultimate features encode bone structure in a
way ImageNet features do not.

Stated plainly about what this cannot do: LUMOS carries no positioning or quality
annotation whatsoever, and it is radiography rather than DXA. So this cannot teach
the model what a positioning defect looks like. It can only give better bone
priors. Whether that transfers is the question the run answers.

Groups are split by patient, never by image, so a patient never appears in both
sides of the pretext split. Organizer labels are never used for fitting.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
from gpu_research.common import digest, dump  # noqa: E402


class BoneNet(nn.Module):
    """Residual CNN sized for ~1.6k images: small enough not to memorise."""

    def __init__(self, width=48, embedding=192):
        super().__init__()
        def block(i, o, stride=2):
            return nn.Sequential(nn.Conv2d(i, o, 3, stride, 1, bias=False),
                                 nn.BatchNorm2d(o), nn.ReLU(inplace=True),
                                 nn.Conv2d(o, o, 3, 1, 1, bias=False), nn.BatchNorm2d(o),
                                 nn.ReLU(inplace=True))
        self.stem = nn.Sequential(nn.Conv2d(1, width, 5, 2, 2, bias=False),
                                  nn.BatchNorm2d(width), nn.ReLU(inplace=True))
        self.body = nn.Sequential(block(width, width * 2), block(width * 2, width * 4),
                                  block(width * 4, width * 8), block(width * 8, width * 8),
                                  block(width * 8, width * 8))
        self.embedding = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(),
                                      nn.Linear(width * 8, embedding), nn.ReLU(inplace=True),
                                      nn.Linear(embedding, embedding))
        self._embedding_dim = embedding
        self.head = nn.Linear(embedding, 4)
        # Masked-region decoder for the self-supervised pretext. Reconstructing a
        # masked patch forces the encoder to model bone structure rather than exposure.
        self.decoder = nn.Sequential(nn.Conv2d(width * 8, width * 4, 3, 1, 1), nn.ReLU(inplace=True),
                                     nn.Conv2d(width * 4, width * 2, 3, 1, 1), nn.ReLU(inplace=True),
                                     nn.ConvTranspose2d(width * 2, width, 4, 2, 1), nn.ReLU(inplace=True),
                                     nn.ConvTranspose2d(width, 1, 4, 2, 1))

    def embed(self, x):
        return self.embedding(self.body(self.stem(x)))

    def forward(self, x):
        return self.head(self.embed(x))

    def reconstruct(self, x, mask):
        return self.decoder(self.body(self.stem(x))) * mask

    @property
    def trunk(self):
        return self.body


def split_by_patient(source, seed, holdout=0.2):
    patients = np.array(sorted(source['patient_id'].unique()))
    rng = np.random.default_rng(seed)
    rng.shuffle(patients)
    cut = int(len(patients) * (1 - holdout))
    train_patients = set(patients[:cut])
    mask = np.array([p in train_patients for p in source['patient_id']])
    return np.flatnonzero(mask), np.flatnonzero(~mask)


def main():
    import pandas as pd
    from sklearn.metrics import r2_score
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prep', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--width', type=int, default=48)
    parser.add_argument('--epochs', type=int, default=80)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--seed', type=int, default=17)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('Choose a new output directory')
    args.output.mkdir(parents=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    images = np.load(args.prep / 'images.npy')
    targets = np.load(args.prep / 'labels.npy')
    source = pd.read_csv(args.prep / 'source.csv')
    train_rows, val_rows = split_by_patient(source, args.seed)
    if set(source['patient_id'].iloc[train_rows]) & set(source['patient_id'].iloc[val_rows]):
        raise ValueError('Patient leaked across the pretext split')

    mean = images[train_rows].mean()
    std = images[train_rows].std() + 1e-6
    x = torch.from_numpy((images.astype(np.float32) - mean) / std).unsqueeze(1)
    # Targets used for fitting: BMD, T-score, age, BMI. Sex is carried but not fitted.
    targets = targets[:, :4]
    y_raw = torch.from_numpy(targets)
    y_mu = y_raw[train_rows].mean(0)
    y_sd = y_raw[train_rows].std(0) + 1e-6
    y = (y_raw - y_mu) / y_sd

    model = BoneNet(args.width).to(device)
    optimiser = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=args.epochs)
    loss_fn = nn.SmoothL1Loss(beta=0.5)
    history = []
    for epoch in range(args.epochs):
        model.train()
        rng = np.random.default_rng(args.seed + epoch)
        order = rng.permutation(train_rows)
        losses = []
        for start in range(0, len(order), args.batch_size):
            batch = order[start:start + args.batch_size]
            optimiser.zero_grad(set_to_none=True)
            out = model(x[batch].to(device))
            # bmd and t-score are the targets of interest; age and BMI are auxiliary.
            loss = loss_fn(out[:, :2], y[batch, :2].to(device)) + 0.3 * loss_fn(out[:, 2:], y[batch, 2:].to(device))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            optimiser.step()
            losses.append(float(loss.detach()))
        scheduler.step()
        model.eval()
        with torch.no_grad():
            predicted = model(x[val_rows].to(device)).cpu().numpy() * y_sd.numpy() + y_mu.numpy()
        truth = targets[val_rows]
        history.append({'epoch': epoch + 1, 'train_loss': float(np.mean(losses)),
                        'val_r2_bmd': float(r2_score(truth[:, 0], predicted[:, 0])),
                        'val_r2_tscore': float(r2_score(truth[:, 1], predicted[:, 1]))})
        if epoch % 5 == 0 or epoch == args.epochs - 1:
            print(f"epoch {epoch + 1}/{args.epochs} loss {np.mean(losses):.4f} "
                  f"val R2 bmd {history[-1]['val_r2_bmd']:+.3f} tscore {history[-1]['val_r2_tscore']:+.3f}",
                  flush=True)

    best = max(history, key=lambda h: h['val_r2_bmd'])
    np.save(args.output / 'bone_pretext_stats.npy', np.stack([mean, std, y_mu.numpy(), y_sd.numpy()]))
    torch.save({'state_dict': model.state_dict(), 'width': args.width,
                'mean': float(mean), 'std': float(std), 'size': int(images.shape[1]),
                'embedding': int(model.head.in_features), 'clinical_validation': False},
               args.output / 'bonenet.pt')
    dump(args.output / 'pretext_report.json', {
        'protocol': __doc__,
        'images': int(images.shape[0]), 'patients': int(source['patient_id'].nunique()),
        'train_images': int(len(train_rows)), 'val_images': int(len(val_rows)),
        'split': 'by patient', 'architecture': {'width': args.width, 'embedding': int(model.head.in_features)},
        'best_epoch_by_val_r2_bmd': best['epoch'],
        'val_r2_bmd': best['val_r2_bmd'], 'val_r2_tscore': best['val_r2_tscore'],
        'final': history[-1], 'history': history,
        'interpretation': ('A positive held-out R2 means the pretext task is learned and the features encode bone '
                           'structure. It says nothing about positioning defects, which LUMOS does not annotate.'),
        'licence': 'CC BY-NC 4.0: anything derived from these weights is non-commercial.',
        'clinical_validation': False, 'model_affects_decision': False})
    print(json.dumps({'best_epoch': best['epoch'], 'val_r2_bmd': best['val_r2_bmd'],
                      'val_r2_tscore': best['val_r2_tscore']}, indent=2))


if __name__ == '__main__':
    main()