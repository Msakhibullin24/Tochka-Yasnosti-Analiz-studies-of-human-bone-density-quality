"""Study-held-out multi-label foundation QC fine-tuning; research adapters only.

Each outer fold has a disjoint inner validation set. Epochs and thresholds are
selected there; outer studies never enter optimization. Routing and measured
axis are reused from the historical baseline, so this is conditional QC, not
end-to-end anatomical validation. No automatic release export/promotion.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedGroupKFold
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gpu_research.common import (CATALOG, HERE, checkpoint_identity, declared_patch_size, digest, dump,
                                 fingerprint, image_input, load_encoder, representations)
from dxaqc.dicom_io import read_any
from dxaqc.decision import decide
from dxaqc.model import CRITERIA, best_f1_threshold, group_of, official_violation_type
from source_integrity import inspect_sources
from train_specialist import OUTPUTS, targets_for
from experiments.nested_criterion_upgrade import validate_outer
from experiments.verify_metric_candidate import compare
from evaluate_organizer_dataset import evaluate


def inner_split(labels, training, seed):
    """Fixed inner split; no model selection against outer labels."""
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


def class_weights(targets, train):
    known = np.isfinite(targets[train])
    positive = np.sum((targets[train] == 1) & known, axis=0)
    negative = np.sum((targets[train] == 0) & known, axis=0)
    supported = (positive > 0) & (negative > 0)
    return np.clip(negative / np.maximum(positive, 1), .25, 10).astype(np.float32), supported


def masked_bce(logits, targets, weights, supported):
    known = torch.isfinite(targets) & supported[None]
    if not known.any():
        raise ValueError('Batch has no supported labels')
    loss = nn.functional.binary_cross_entropy_with_logits(
        logits.float(), torch.nan_to_num(targets), pos_weight=weights, reduction='none')
    return loss[known].mean()


class FoundationQC(nn.Module):
    def __init__(self, encoder, kind, pooling, dimension, last_blocks):
        super().__init__()
        self.encoder, self.kind, self.pooling = encoder, kind, pooling
        self.encoder.requires_grad_(False)
        if last_blocks:
            blocks = encoder.encoder.layers if kind == 'siglip' else encoder.model.layer
            if not 0 < last_blocks <= len(blocks):
                raise ValueError('Invalid number of trainable transformer blocks')
            for block in blocks[-last_blocks:]:
                block.requires_grad_(True)
        self.head = nn.Linear(dimension, len(OUTPUTS))

    def forward(self, values):
        vector, _ = representations(self.encoder, values, self.kind, self.pooling)
        return self.head(vector.float())


def run(args):
    if (args.output.exists() or args.epochs < 1 or args.batch_size < 1 or args.last_blocks < 0
            or args.patience < 1 or args.head_lr <= 0 or args.encoder_lr <= 0):
        raise ValueError('Choose a new output directory and valid training parameters')
    if not torch.cuda.is_available() and args.device == 'cuda':
        raise RuntimeError('CUDA requested but unavailable')
    if args.precision == 'bf16' and (args.device != 'cuda' or not torch.cuda.is_bf16_supported()):
        raise ValueError('BF16 training needs a supported CUDA GPU; use fp32 for CPU smoke')
    if args.device == 'cpu' and args.last_blocks:
        raise ValueError('CPU mode is for frozen-head smoke tests only')
    entry = CATALOG[args.model]
    size = args.size or entry['size']
    patch = declared_patch_size(args.model_dir, entry['kind'])
    if size % patch or not 32 <= size <= 2048:
        raise ValueError(f'Input size must be divisible by patch size {patch} and within 32..2048')
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
    fold_reports = []
    start = time.perf_counter()
    torch.set_num_threads(4)
    # Fold subset reports are never treated as full OOF candidates.
    folds = sorted(baseline.fold.unique()) if args.fold is None else [args.fold]
    if not set(folds) <= set(baseline.fold):
        raise ValueError('Unknown outer fold')
    for fold in folds:
        torch.manual_seed(args.seed)
        if args.device == 'cuda':
            torch.cuda.manual_seed_all(args.seed)
            torch.cuda.reset_peak_memory_stats()
        held = set(baseline.loc[baseline.fold == fold, 'study'].astype(str))
        training = np.flatnonzero(~np.isin(groups, list(held)))
        testing = baseline.loc[baseline.fold == fold, 'index'].to_numpy(dtype=int)
        train, val = inner_split(labels, training, args.seed + 100)
        if (set(groups[train]) | set(groups[val])) & set(groups[testing]):
            raise ValueError('Outer study leakage')
        # FP32 master parameters for optimizer stability; BF16 autocast is independent.
        processor, encoder = load_encoder(args.model_dir, entry['kind'], args.device, 'fp32')
        images = []
        for row in labels.itertuples():
            image = read_any(args.dataset / row.first_source_path)
            if image.pixel_sha256 != audited[row.first_source_path]:
                raise ValueError('Source pixels changed after audit')
            pixels = np.ascontiguousarray(image.pixels[:, ::-1] if row.region == 'hip_left' else image.pixels)
            values, _ = image_input(processor, pixels, size, entry['kind'])
            images.append(values[0])
        # Inputs stay in CPU RAM; training batches alone are copied to VRAM.
        x = torch.stack(images)
        with torch.no_grad():
            vector, _ = representations(encoder, x[:1], entry['kind'], args.pooling)
        model = FoundationQC(encoder, entry['kind'], args.pooling, vector.shape[1], args.last_blocks).to(args.device)
        if args.last_blocks and hasattr(encoder, 'gradient_checkpointing_enable'):
            encoder.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
        head_parameters = list(model.head.parameters())
        encoder_parameters = [p for p in encoder.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW([
            {'params': head_parameters, 'lr': args.head_lr},
            {'params': encoder_parameters, 'lr': args.encoder_lr}], weight_decay=.01)
        weight, supported = class_weights(targets, train)
        weights = torch.tensor(weight, device=args.device)
        enabled = torch.tensor(supported, device=args.device)
        y = torch.tensor(targets, device=args.device)
        trainable = {name for name, p in model.named_parameters() if p.requires_grad}
        best, best_epoch, best_loss = None, None, float('inf')
        history = []
        rng = np.random.default_rng(args.seed)

        def predict(indices):
            model.eval()
            values = []
            with torch.no_grad():
                for pos in range(0, len(indices), args.batch_size):
                    batch = indices[pos:pos + args.batch_size]
                    with torch.autocast(args.device, dtype=torch.bfloat16, enabled=args.precision == 'bf16'):
                        values.append(model(x[batch].to(args.device)).float())
            return torch.cat(values)

        for epoch in range(args.epochs):
            model.train()
            if not args.last_blocks:
                encoder.eval()
            losses = []
            order = rng.permutation(train)
            for pos in range(0, len(order), args.batch_size):
                batch = order[pos:pos + args.batch_size]
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(args.device, dtype=torch.bfloat16, enabled=args.precision == 'bf16'):
                    loss = masked_bce(model(x[batch].to(args.device)), y[batch], weights, enabled)
                if not torch.isfinite(loss):
                    raise ValueError('Nonfinite loss')
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
                optimizer.step()
                losses.append(float(loss.detach()))
            val_logits = predict(val)
            val_loss = float(masked_bce(val_logits, y[val], weights, enabled))
            history.append({'epoch': epoch + 1, 'training_loss': float(np.mean(losses)), 'validation_loss': val_loss})
            if val_loss < best_loss:
                best_loss, best_epoch = val_loss, epoch + 1
                best = {name: value.detach().cpu().clone() for name, value in model.state_dict().items() if name in trainable}
            print(f'fold={fold} epoch={epoch+1} val_loss={val_loss:.5f}', flush=True)
            if epoch + 1 - best_epoch >= args.patience:
                break
        model.load_state_dict(best, strict=False)
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
            scores = {name: float(test_scores[position, OUTPUTS.index(name)]) for name in CRITERIA[group]}
            angle = json.loads(baseline.at[index, 'criterion_states']).get('spine_axis', {}).get('angle_deg')
            decision = decide(group, float(test_scores[position, 0]), scores,
                              {'spine_abs_angle_deg': angle}, float(np.nextafter(1., np.inf)), thresholds)
            for key, value in {
                'quality_score': float(test_scores[position, 0]), 'quality_pred': decision['quality'],
                'violation_type': official_violation_type(decision['violations']),
                'criterion_scores': json.dumps(scores), 'criterion_states': json.dumps(decision['criterion_states']),
                'criterion_thresholds': json.dumps(thresholds), 'decision_version': 'research-foundation-finetune-1',
                'decision_reason': decision['decision_reason']}.items():
                candidate.at[index, key] = value
        fold_dir = args.output / f'fold-{fold}'
        fold_dir.mkdir()
        torch.save({'state_dict': best, 'outputs': list(OUTPUTS), 'thresholds': thresholds,
                    'model': args.model, 'pooling': args.pooling, 'size': size, 'last_blocks': args.last_blocks,
                    'base_checkpoint': identity, 'clinical_validation': False}, fold_dir / 'adapter.pt')
        report = {'fold': int(fold), 'training_studies': sorted(set(groups[train])),
                  'validation_studies': sorted(set(groups[val])), 'test_studies': sorted(set(groups[testing])),
                  'best_epoch': best_epoch, 'unvalidated_thresholds': unvalidated,
                  'history': history, 'thresholds': thresholds, 'adapter_sha256': digest(fold_dir / 'adapter.pt'),
                  'peak_vram_bytes': torch.cuda.max_memory_allocated() if args.device == 'cuda' else None}
        dump(fold_dir / 'training.json', report)
        fold_reports.append(report)
        del model, encoder, optimizer, best, x, images
        if args.device == 'cuda':
            torch.cuda.empty_cache()
    partial = args.fold is not None
    path = args.output / ('partial_predictions.csv' if partial else 'candidate_oof.csv')
    candidate.loc[candidate.fold.isin(folds)].to_csv(path, index=False)
    report = {'protocol': __doc__, 'folds': fold_reports, 'partial': partial,
              'base_checkpoint': identity, 'source_fingerprint': fingerprint(audit['entries']),
              'labels_sha256': digest(args.labels), 'baseline_sha256': digest(args.baseline),
              'code_sha256': {'finetune': digest(__file__), 'common': digest(Path(__file__).with_name('common.py'))},
              'elapsed_seconds': time.perf_counter() - start,
              'clinical_validation': False, 'model_affects_decision': False,
              'limitations': ['Previously inspected cohort; patient independence unverified.',
                              'Baseline routing and axes reused; no new anatomical accuracy measured.',
                              'Small validation sets can lack positive rare criteria; such thresholds are disabled.',
                              'Raw sigmoid scores are uncalibrated; calibration needs separate validation data.',
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
    p.add_argument('--labels', type=Path, default=HERE / 'labels/image_labels.csv')
    p.add_argument('--baseline', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--device', choices=('cuda', 'cpu'), default='cuda')
    p.add_argument('--precision', choices=('bf16', 'fp32'), default='bf16')
    p.add_argument('--size', type=int)
    p.add_argument('--pooling', choices=('global', 'global-mean'), default='global-mean')
    p.add_argument('--last-blocks', type=int, default=2)
    p.add_argument('--batch-size', type=int, default=2)
    p.add_argument('--epochs', type=int, default=30)
    p.add_argument('--patience', type=int, default=5)
    p.add_argument('--seed', type=int, default=17)
    p.add_argument('--head-lr', type=float, default=.001)
    p.add_argument('--encoder-lr', type=float, default=.00001)
    p.add_argument('--fold', type=int, help='Single-fold smoke; produces partial report, never full OOF')
    run(p.parse_args())


if __name__ == '__main__':
    main()
