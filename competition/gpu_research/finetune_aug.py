"""Augmented, narrowly-unfrozen fine-tuning of a frozen foundation encoder.

The tail-2 block fine-tune in finetune.py memorised each ~200-image training
fold. This trainer keeps the same outer folds, inner split, routing and decision
layer, but changes the optimisation recipe:

  * GPU augmentation (rotation, translation, scale, contrast, brightness, noise)
    so the encoder never sees the same frame twice;
  * narrow trainable scope - head only, head + LayerNorm, or head + LayerNorm +
    the last N blocks - instead of two whole blocks;
  * cosine schedule with warmup and a much smaller encoder learning rate;
  * epoch selection on the inner validation split only.

Weights stay uint8 in host RAM and are normalised per batch on the GPU.
Research only; adapters are never promoted automatically.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedGroupKFold
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gpu_research.common import (CATALOG, checkpoint_identity, digest, dump, fingerprint,
                                 image_input, load_encoder, representations)
from dxaqc.dicom_io import read_any
from dxaqc.decision import decide
from dxaqc.model import CRITERIA, best_f1_threshold, group_of, official_violation_type
from source_integrity import inspect_sources
from train_specialist import OUTPUTS, targets_for
from experiments.nested_criterion_upgrade import validate_outer
from experiments.verify_metric_candidate import compare
from evaluate_organizer_dataset import evaluate

TRAIN_SCOPES = ('head', 'head_ln', 'head_ln_blocks')


class Scope(torch.nn.Module):
    """Encoder with a trainable-parameter mask chosen by the requested scope."""

    def __init__(self, encoder, kind, last_blocks):
        super().__init__()
        self.encoder, self.kind = encoder, kind
        self.encoder.requires_grad_(False)
        blocks = encoder.encoder.layers if kind == 'siglip' else encoder.model.layer
        self.last_blocks = 0
        if last_blocks:
            if not 0 < last_blocks <= len(blocks):
                raise ValueError('Invalid number of trainable blocks')
            self.last_blocks = last_blocks
            for block in blocks[-last_blocks:]:
                block.requires_grad_(True)
        self.layernorm = last_blocks == 0

    def trainable(self, scope):
        if self.layernorm:
            for module in self.encoder.modules():
                if isinstance(module, nn.LayerNorm):
                    module.requires_grad_(True)
        named = [(n, p) for n, p in self.encoder.named_parameters() if p.requires_grad]
        return named


def augment(batch, generator, rotation_deg, translate_frac, scale_jitter,
            contrast, brightness, noise):
    """Random affine plus intensity jitter, done on the GPU in float32."""
    n = batch.shape[0]
    device = batch.device
    angle = (torch.rand(n, device=device, generator=generator) * 2 - 1) * math.radians(rotation_deg)
    scale = 1.0 + (torch.rand(n, device=device, generator=generator) * 2 - 1) * scale_jitter
    tx = (torch.rand(n, device=device, generator=generator) * 2 - 1) * translate_frac
    ty = (torch.rand(n, device=device, generator=generator) * 2 - 1) * translate_frac
    cos, sin = torch.cos(angle) / scale, torch.sin(angle) / scale
    theta = torch.zeros(n, 2, 3, device=device)
    theta[:, 0, 0], theta[:, 0, 1], theta[:, 0, 2] = cos, -sin, tx
    theta[:, 1, 0], theta[:, 1, 1], theta[:, 1, 2] = sin, cos, ty
    grid = torch.nn.functional.affine_grid(theta, batch.shape, align_corners=False)
    out = torch.nn.functional.grid_sample(batch, grid, mode='bilinear',
                                          padding_mode='zeros', align_corners=False)
    contrast = 1.0 + (torch.rand(n, 1, 1, 1, device=device, generator=generator) * 2 - 1) * contrast
    brightness = (torch.rand(n, 1, 1, 1, device=device, generator=generator) * 2 - 1) * brightness
    mean = out.mean(dim=(1, 2, 3), keepdim=True)
    out = (out - mean) * contrast + mean + brightness
    if noise:
        out = out + torch.randn(out.shape, device=device, generator=generator) * noise
    return out


def masked_bce(logits, targets, weights, supported):
    known = torch.isfinite(targets) & supported[None]
    if not known.any():
        raise ValueError('Batch has no supported labels')
    loss = nn.functional.binary_cross_entropy_with_logits(
        logits.float(), torch.nan_to_num(targets), pos_weight=weights, reduction='none')
    return loss[known].mean()


def class_weights(targets, train):
    known = np.isfinite(targets[train])
    positive = np.sum((targets[train] == 1) & known, axis=0)
    negative = np.sum((targets[train] == 0) & known, axis=0)
    supported = (positive > 0) & (negative > 0)
    return np.clip(negative / np.maximum(positive, 1), .25, 10).astype(np.float32), supported


def inner_split(labels, training, seed):
    groups = labels.study_key.astype(str).to_numpy()
    y = labels.quality_class.to_numpy()
    tr, va = next(StratifiedGroupKFold(4, shuffle=True, random_state=seed).split(
        training, y[training], groups[training]))
    train, validation = training[tr], training[va]
    if set(groups[train]) & set(groups[validation]):
        raise ValueError('Inner study leakage')
    if any(len(np.unique(y[part])) < 2 for part in (train, validation)):
        raise ValueError('Inner partitions require both quality classes')
    return train, validation


def run(args):
    if args.output.exists() or args.scope not in TRAIN_SCOPES:
        raise ValueError('Choose a new output directory and a valid scope')
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required')
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError('BF16 is required for this trainer')
    entry = CATALOG[args.model]
    size = args.size or entry['size']
    if size % 16 or not 32 <= size <= 2048:
        raise ValueError('Input size must be compatible with the patch grid')
    labels = pd.read_csv(args.labels)
    labels = labels[labels.quality_class.notna()].reset_index(drop=True)
    baseline = pd.read_csv(args.baseline).sort_values('index').reset_index(drop=True)
    validate_outer(labels, baseline)
    audit = inspect_sources([('organiser', args.labels, args.dataset)])
    audited = {r['path']: r['pixel_sha256_current'] for r in audit['entries']}
    identity = checkpoint_identity(args.model_dir, entry['kind'])
    targets = targets_for(labels)
    args.output.mkdir(parents=True)
    dump(args.output / 'config.json', {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()})
    candidate = baseline.copy()
    groups = labels.study_key.astype(str).to_numpy()
    folds = sorted(baseline.fold.unique()) if args.fold is None else [args.fold]
    if not set(folds) <= set(baseline.fold):
        raise ValueError('Unknown outer fold')
    start = time.perf_counter()
    torch.set_num_threads(4)

    processor, encoder = load_encoder(args.model_dir, entry['kind'], args.device, 'fp32')
    mean = torch.tensor(processor.image_mean, device=args.device).view(1, 3, 1, 1)
    std = torch.tensor(processor.image_std, device=args.device).view(1, 3, 1, 1)
    cache = []
    for row in labels.itertuples():
        image = read_any(args.dataset / row.first_source_path)
        if image.pixel_sha256 != audited[row.first_source_path]:
            raise ValueError('Source pixels changed after audit')
        pixels = np.ascontiguousarray(image.pixels[:, ::-1] if row.region == 'hip_left' else image.pixels)
        values, _ = image_input(processor, pixels, size, entry['kind'])
        cache.append(values[0])
    x = torch.stack(cache).mul(255).round().clamp(0, 255).to(torch.uint8)
    del cache

    fold_reports = []
    for fold in folds:
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        torch.cuda.reset_peak_memory_stats()
        held = set(baseline.loc[baseline.fold == fold, 'study'].astype(str))
        training = np.flatnonzero(~np.isin(groups, list(held)))
        testing = baseline.loc[baseline.fold == fold, 'index'].to_numpy(dtype=int)
        train, val = inner_split(labels, training, args.seed + 100)
        if (set(groups[train]) | set(groups[val])) & set(groups[testing]):
            raise ValueError('Outer study leakage')

        scope = Scope(encoder, entry['kind'], args.last_blocks if args.scope == 'head_ln_blocks' else 0)
        encoder_parameters = scope.trainable(args.scope)
        head = nn.Linear(int(args.head_dim), len(OUTPUTS)).to(args.device)
        parameters = [{'params': list(head.parameters()), 'lr': args.head_lr}]
        if encoder_parameters:
            parameters.append({'params': [p for _, p in encoder_parameters], 'lr': args.encoder_lr})
        optimizer = torch.optim.AdamW(parameters, weight_decay=.01)
        weight, supported = class_weights(targets, train)
        weights = torch.tensor(weight, device=args.device)
        enabled = torch.tensor(supported, device=args.device)
        y = torch.tensor(targets, device=args.device)
        trainable = {n for n, p in head.named_parameters()}
        trainable |= {f'encoder.{n}' for n, p in encoder_parameters}
        steps = max(1, math.ceil(len(train) / args.batch_size)) * args.epochs
        warmup = max(1, int(0.1 * steps))
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lambda s: (s + 1) / warmup if s < warmup else
            0.5 * (1 + math.cos(math.pi * (s - warmup) / max(1, steps - warmup))))
        rng = np.random.default_rng(args.seed)
        gen = torch.Generator(device=args.device).manual_seed(args.seed)
        best, best_epoch, best_loss = None, None, float('inf')
        history = []

        def forward(indices, training_mode):
            raw = x[indices].to(args.device, non_blocking=True).float().div(255)
            if training_mode:
                raw = augment(raw, gen, args.rotation_deg, args.translate_frac, args.scale_jitter,
                              args.contrast, args.brightness, args.noise)
            values = (raw - mean) / std
            with torch.autocast('cuda', dtype=torch.bfloat16):
                vector, _ = representations(encoder, values, entry['kind'], args.pooling)
            return head(vector.float())

        def predict(indices):
            encoder.eval()
            head.eval()
            out = []
            with torch.no_grad():
                for pos in range(0, len(indices), args.eval_batch_size):
                    out.append(forward(indices[pos:pos + args.eval_batch_size], False).float())
            return torch.cat(out)

        for epoch in range(args.epochs):
            head.train()
            if encoder_parameters:
                encoder.train()
            losses = []
            order = rng.permutation(train)
            for pos in range(0, len(order), args.batch_size):
                batch = order[pos:pos + args.batch_size]
                optimizer.zero_grad(set_to_none=True)
                loss = masked_bce(forward(batch, True), y[batch], weights, enabled)
                if not torch.isfinite(loss):
                    raise ValueError('Nonfinite loss')
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    [p for group in optimizer.param_groups for p in group['params']], 1., error_if_nonfinite=True)
                optimizer.step()
                scheduler.step()
                losses.append(float(loss.detach()))
            val_loss = float(masked_bce(predict(val), y[val], weights, enabled))
            history.append({'epoch': epoch + 1, 'training_loss': float(np.mean(losses)),
                            'validation_loss': val_loss,
                            'head_lr': scheduler.get_last_lr()[0]})
            if val_loss < best_loss:
                best_loss, best_epoch = val_loss, epoch + 1
                state = {n: v.detach().cpu().clone() for n, v in head.state_dict().items()}
                state |= {f'encoder.{n}': v.detach().cpu().clone() for n, v in encoder_parameters}
                best = state
            print(f'fold={fold} epoch={epoch+1} val_loss={val_loss:.5f}', flush=True)
            if epoch + 1 - best_epoch >= args.patience:
                break
        head.load_state_dict({k: v for k, v in best.items() if not k.startswith('encoder.')})
        missing = [k for k in best if k.startswith('encoder.')]
        if missing:
            own = dict(encoder.named_parameters())
            for key in missing:
                own[key[len('encoder.'):]].data.copy_(best[key])
        val_scores = torch.sigmoid(predict(val)).cpu().numpy()
        test_scores = torch.sigmoid(predict(testing)).cpu().numpy()
        thresholds, unvalidated = {}, []
        for col, name in enumerate(OUTPUTS[1:], 1):
            known = np.isfinite(targets[val, col])
            if supported[col] and len(np.unique(targets[val, col][known])) == 2:
                thresholds[name] = float(best_f1_threshold(targets[val, col][known].astype(int), val_scores[known, col]))
            else:
                thresholds[name] = float(np.nextafter(1., np.inf))
                unvalidated.append(name)
        for position, index in enumerate(testing):
            group = group_of(baseline.at[index, 'predicted_region'])
            scores = {n: float(test_scores[position, OUTPUTS.index(n)]) for n in CRITERIA[group]}
            angle = json.loads(baseline.at[index, 'criterion_states']).get('spine_axis', {}).get('angle_deg')
            decision = decide(group, float(test_scores[position, 0]), scores,
                              {'spine_abs_angle_deg': angle}, float(np.nextafter(1., np.inf)), thresholds)
            for key, value in {'quality_score': float(test_scores[position, 0]),
                               'quality_pred': decision['quality'],
                               'violation_type': official_violation_type(decision['violations']),
                               'criterion_scores': json.dumps(scores),
                               'criterion_states': json.dumps(decision['criterion_states']),
                               'criterion_thresholds': json.dumps(thresholds),
                               'decision_version': 'research-augmented-finetune-1',
                               'decision_reason': decision['decision_reason']}.items():
                candidate.at[index, key] = value
        fold_dir = args.output / f'fold-{fold}'
        fold_dir.mkdir()
        torch.save({'state_dict': best, 'outputs': list(OUTPUTS), 'thresholds': thresholds,
                    'model': args.model, 'scope': args.scope, 'pooling': args.pooling, 'size': size,
                    'base_checkpoint': identity, 'clinical_validation': False}, fold_dir / 'adapter.pt')
        dump(fold_dir / 'training.json',
             {'fold': int(fold), 'best_epoch': best_epoch, 'scope': args.scope,
              'trainable_encoder_tensors': len(encoder_parameters),
              'training_studies': sorted(set(groups[train])),
              'validation_studies': sorted(set(groups[val])),
              'test_studies': sorted(set(groups[testing])),
              'unvalidated_thresholds': unvalidated, 'history': history, 'thresholds': thresholds,
              'adapter_sha256': digest(fold_dir / 'adapter.pt'),
              'peak_vram_bytes': torch.cuda.max_memory_allocated()})
        fold_reports.append({'fold': int(fold), 'best_epoch': best_epoch,
                             'epochs_run': len(history),
                             'unvalidated_thresholds': unvalidated,
                             'adapter_sha256': digest(fold_dir / 'adapter.pt'),
                             'peak_vram_bytes': torch.cuda.max_memory_allocated()})
        head.zero_grad(set_to_none=True)
        del head, scope
        torch.cuda.empty_cache()
    partial = args.fold is not None
    path = args.output / ('partial_predictions.csv' if partial else 'candidate_oof.csv')
    candidate.loc[candidate.fold.isin(folds)].to_csv(path, index=False)
    report = {'protocol': __doc__, 'augmentation': {
        'rotation_deg': args.rotation_deg, 'translate_frac': args.translate_frac,
        'scale_jitter': args.scale_jitter, 'contrast': args.contrast,
        'brightness': args.brightness, 'noise': args.noise},
        'scope': args.scope, 'folds': fold_reports, 'partial': partial,
        'base_checkpoint': identity, 'source_fingerprint': fingerprint(audit['entries']),
        'labels_sha256': digest(args.labels), 'baseline_sha256': digest(args.baseline),
        'code_sha256': {'augmented': digest(__file__),
                        'common': digest(Path(__file__).with_name('common.py'))},
        'elapsed_seconds': time.perf_counter() - start,
        'clinical_validation': False, 'model_affects_decision': False,
        'limitations': ['Previously inspected cohort; patient independence unverified.',
                        'Baseline routing and axes reused; no new anatomical accuracy measured.',
                        'Augmentation is image-space only; it cannot create new anatomy or new defect types.',
                        'Small validation sets can lack positive rare criteria; such thresholds are disabled.',
                        'Raw sigmoid scores are uncalibrated.',
                        'Multiple runs against this cohort are exploratory, not an independent test.']}
    if not partial:
        report['metrics'] = evaluate(args.labels, path, repeats=2000)
        report['screen'] = compare(args.baseline, path, args.labels)
    dump(args.output / 'evaluation.json', report)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', choices=CATALOG, required=True)
    p.add_argument('--model-dir', type=Path, required=True)
    p.add_argument('--dataset', type=Path, required=True)
    p.add_argument('--labels', type=Path, default=Path(__file__).resolve().parents[1] / 'labels/image_labels.csv')
    p.add_argument('--baseline', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--scope', choices=TRAIN_SCOPES, default='head_ln')
    p.add_argument('--device', choices=('cuda', 'cpu'), default='cuda')
    p.add_argument('--last-blocks', type=int, default=1)
    p.add_argument('--pooling', choices=('global', 'global-mean'), default='global-mean')
    p.add_argument('--size', type=int)
    p.add_argument('--head-dim', type=int, default=2048, help='DINOv3-L pooled+patch-mean width')
    p.add_argument('--batch-size', type=int, default=8)
    p.add_argument('--eval-batch-size', type=int, default=16)
    p.add_argument('--epochs', type=int, default=40)
    p.add_argument('--patience', type=int, default=8)
    p.add_argument('--head-lr', type=float, default=.001)
    p.add_argument('--encoder-lr', type=float, default=1e-5)
    p.add_argument('--rotation-deg', type=float, default=8.)
    p.add_argument('--translate-frac', type=float, default=.05)
    p.add_argument('--scale-jitter', type=float, default=.05)
    p.add_argument('--contrast', type=float, default=.2)
    p.add_argument('--brightness', type=float, default=.1)
    p.add_argument('--noise', type=float, default=.02)
    p.add_argument('--seed', type=int, default=17)
    p.add_argument('--fold', type=int)
    run(p.parse_args())


if __name__ == '__main__':
    main()
