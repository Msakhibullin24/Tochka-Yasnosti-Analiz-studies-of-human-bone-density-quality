"""Train a frozen-backbone binary/multilabel QC candidate; export offline shadow inference.

Train/validation/test studies are disjoint; validation chooses thresholds only.
The held-out report is an internal experiment, not an external clinical validation.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedGroupKFold

from dxaqc.dicom_io import read_any
from dxaqc.model import CRITERIA, best_f1_threshold, group_of
from dxaqc.specialist_qc import (OUTPUTS, SCHEMA_VERSION, PREPROCESS_VERSION,
                               ImageQC, MultiViewQC, create_encoder, digest, image_tensor, masked_loss)
from experiments.cnn_quality import valid_labels
from train import binary_metrics


def targets_for(labels: pd.DataFrame) -> np.ndarray:
    target = np.full((len(labels), len(OUTPUTS)), np.nan, np.float32)
    target[:, 0] = labels.quality_class.to_numpy()
    for i, name in enumerate(OUTPUTS[1:], 1):
        applicable = np.array([name in CRITERIA[group_of(region)] for region in labels.region])
        values = pd.to_numeric(labels[name], errors='raise').to_numpy(dtype=np.float32)
        if np.any(np.isfinite(values) & ~np.isin(values, [0, 1])) or np.isinf(values).any():
            raise ValueError(f'Invalid labels for {name}')
        target[applicable, i] = values[applicable]
    return target


def split_indices(labels: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = len(labels)
    # One fixed outer fold for test, one fixed inner fold for threshold selection.
    y, groups = labels.quality_class.to_numpy(), labels.study_key.to_numpy()
    outer = StratifiedGroupKFold(5, shuffle=True, random_state=17)
    trainval, test = next(outer.split(np.zeros(n), y, groups))
    inner = StratifiedGroupKFold(4, shuffle=True, random_state=18)
    train, val = next(inner.split(trainval, y[trainval], groups[trainval]))
    partitions = (trainval[train], trainval[val], test)
    for a, b in ((0, 1), (0, 2), (1, 2)):
        if set(groups[partitions[a]]) & set(groups[partitions[b]]):
            raise ValueError('Study leakage')
    if any(len(np.unique(y[part])) != 2 for part in partitions):
        raise ValueError('Each split requires both binary classes')
    return partitions


def resolve_device(requested: str) -> torch.device:
    if requested not in ('cpu', 'cuda', 'auto'):
        raise ValueError('Device must be cpu, cuda, or auto')
    if requested == 'cuda' and not torch.cuda.is_available():
        raise ValueError('CUDA was requested but is unavailable')
    return torch.device('cuda' if requested == 'cuda' or
                        (requested == 'auto' and torch.cuda.is_available()) else 'cpu')


def run(args) -> dict:
    if args.output.exists():
        raise ValueError('Choose a new output directory; experiments are immutable')
    if args.epochs < 1 or not 32 <= args.size <= 1024 or args.threads < 1:
        raise ValueError('Invalid training configuration')
    torch.set_num_threads(args.threads)
    torch.manual_seed(17)
    device = resolve_device(getattr(args, 'device', 'cpu'))
    if device.type == 'cuda':
        torch.cuda.manual_seed_all(17)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    start = time.perf_counter()
    provenance = {name: digest(Path(__file__).parent/name) for name in
                  ('train_specialist.py','dxaqc/specialist_qc.py','dxaqc/dicom_io.py')}
    labels = valid_labels(args.labels)
    targets = targets_for(labels)
    train, val, test = split_indices(labels)
    region_scope = getattr(args, 'region', 'all')
    train_mode = getattr(args, 'train_mode', 'head')
    views = getattr(args, 'views', 'full')
    if region_scope not in ('all', 'spine', 'hip') or train_mode not in ('head', 'last-stage') or views not in ('full','full-center'):
        raise ValueError('Unsupported training configuration')
    if train_mode == 'last-stage' and args.backbone != 'convnextv2_tiny':
        raise ValueError('Last-stage training currently requires convnextv2_tiny')
    if region_scope != 'all':
        applicable = np.array([group_of(r) == region_scope for r in labels.region])
        targets[~applicable] = np.nan
        train, val, test = [idx[applicable[idx]] for idx in (train, val, test)]
    if any(len(np.unique(targets[idx,0])) != 2 for idx in (train,val,test)):
        raise ValueError('Regional partitions need both quality classes')
    # Require the exact catalog checkpoint, not merely a similarly named file.
    catalog = json.loads((Path(__file__).resolve().parents[1] / 'docs/competition/specialist_sources.json').read_text())
    artifact_id = 'convnextv2_tiny_weights' if args.backbone == 'convnextv2_tiny' else 'efficientnet_b4_weights'
    expected = next(a['sha256'] for a in catalog['artifacts'] if a['id'] == artifact_id)
    if digest(args.weights) != expected:
        raise ValueError('Backbone checkpoint checksum mismatch')
    encoder = create_encoder(args.backbone, args.weights).to(device)
    encoder.requires_grad_(False)
    if train_mode == 'last-stage':
        prefix = torch.nn.Sequential(encoder.stem, *list(encoder.stages)[:-1]).eval()
        tail = torch.nn.Sequential(encoder.stages[-1], encoder.norm_pre, encoder.head)
        tail.requires_grad_(True)
    else:
        prefix, tail = encoder, torch.nn.Identity()
    features, pixel_groups = [], {}
    changed_hashes = 0
    for i, row in enumerate(labels.itertuples(), 1):
        path = (args.dataset / row.first_source_path).resolve()
        if not path.is_relative_to(args.dataset.resolve()):
            raise ValueError('Image path escapes dataset root')
        img = read_any(path)
        if pixel_groups.setdefault(img.pixel_sha256, row.study_key) != row.study_key:
            raise ValueError('Current pixel duplicate crosses study groups')
        changed_hashes += img.pixel_sha256 != row.pixel_sha256
        if img.pixels.shape != (row.rows, row.columns):
            raise ValueError('Image dimensions changed')
        # Full frame, original orientation. No unlabelled geometric augmentation.
        with torch.no_grad():
            tensor = image_tensor(img.pixels, args.size)[None].to(device)
            tensors = [tensor]
            if views == 'full-center':
                tensors.append(MultiViewQC.center_view(tensor))
            features.append(torch.stack([prefix(t)[0] for t in tensors]))
        if i % 25 == 0:
            print(f'features {i}/{len(labels)}', flush=True)
    x = torch.stack(features)
    y = torch.from_numpy(targets).to(device)
    supported = np.array([len(np.unique(targets[train, i][np.isfinite(targets[train, i])])) == 2
                          for i in range(len(OUTPUTS))])
    if not supported[0]:
        raise ValueError('Training requires both quality classes')
    training_targets = y.clone()
    training_targets[:, ~supported] = float('nan')
    with torch.no_grad():
        feature_dim = tail(x[0,0][None]).shape[1]
    head = torch.nn.Linear(feature_dim*x.shape[1], len(OUTPUTS)).to(device)
    def predict_cached(indices):
        positions = torch.as_tensor(indices, dtype=torch.long, device=device)
        return head(torch.cat([tail(x[positions, v]) for v in range(x.shape[1])], dim=1))
    parameters = [{'params': head.parameters(), 'lr': .003}]
    if train_mode == 'last-stage':
        parameters.append({'params': tail.parameters(), 'lr': 1e-5})
    optimizer = torch.optim.AdamW(parameters, weight_decay=.01)
    rng = np.random.default_rng(17)
    history = []
    best_loss, best_state, best_epoch = float('inf'), None, None
    batch_size = 8 if train_mode == 'last-stage' else 32
    for epoch in range(args.epochs):
        for indices in np.array_split(rng.permutation(train), max(1, int(np.ceil(len(train)/batch_size)))):
            optimizer.zero_grad(set_to_none=True)
            positions = torch.as_tensor(indices, dtype=torch.long, device=device)
            loss = masked_loss(predict_cached(indices), training_targets[positions], args.loss)
            loss.backward()
            if train_mode == 'last-stage':
                torch.nn.utils.clip_grad_norm_([p for group in optimizer.param_groups for p in group['params']], 1.0, error_if_nonfinite=True)
            optimizer.step()
        with torch.no_grad():
            train_positions = torch.as_tensor(train, dtype=torch.long, device=device)
            val_positions = torch.as_tensor(val, dtype=torch.long, device=device)
            train_loss = float(masked_loss(predict_cached(train), training_targets[train_positions], args.loss))
            val_loss = float(masked_loss(predict_cached(val), training_targets[val_positions], args.loss))
        if val_loss < best_loss:
            best_loss, best_epoch = val_loss, epoch+1
            best_state = (copy.deepcopy(head.state_dict()), copy.deepcopy(tail.state_dict()))
        history.append({'epoch': epoch+1, 'train_loss': train_loss, 'validation_loss': val_loss})
        if epoch == 0 or (epoch+1) % 10 == 0:
            print(f'epoch {epoch+1}/{args.epochs} train={train_loss:.5f} val={val_loss:.5f}', flush=True)
    # Preserve historical head-only baseline behavior. Fine-tuning selects on validation only.
    if train_mode == 'last-stage':
        head.load_state_dict(best_state[0]); tail.load_state_dict(best_state[1])
    head.eval(); encoder.eval(); tail.eval()
    with torch.no_grad():
        scores = np.concatenate([torch.sigmoid(predict_cached(idx)).cpu().numpy()
                                 for idx in np.array_split(np.arange(len(x)), max(1,len(x)//8))])
    thresholds, metrics = {}, {}
    for i, name in enumerate(OUTPUTS):
        known_val = val[np.isfinite(targets[val, i])]
        known_test = test[np.isfinite(targets[test, i])]
        threshold = None
        if supported[i] and len(np.unique(targets[known_val, i])) == 2:
            threshold = float(best_f1_threshold(targets[known_val, i].astype(int), scores[known_val, i]))
        thresholds[name] = threshold
        metrics[name] = {'training_supported': bool(supported[i]), 'validation_known': len(known_val),
                         'test_known': len(known_test), 'threshold': threshold}
        if threshold is not None and len(known_test):
            metrics[name].update(binary_metrics(targets[known_test, i].astype(int), scores[known_test, i],
                                                (scores[known_test, i] >= threshold).astype(int)))
            if not np.any(targets[known_test, i] == 1):
                metrics[name]['sensitivity'] = None
                metrics[name]['balanced_accuracy'] = None
            if not np.any(targets[known_test, i] == 0):
                metrics[name]['specificity'] = None
                metrics[name]['balanced_accuracy'] = None
    args.output.mkdir(parents=True)
    model = (ImageQC(encoder, head) if views == 'full' else MultiViewQC(encoder, head)).cpu().eval()
    example = torch.zeros(1, 3, args.size, args.size)
    traced = torch.jit.trace(model, example)
    traced.save(str(args.output / 'model.ts'))
    from safetensors.torch import save_file
    save_file(head.state_dict(), str(args.output / 'head.safetensors'))
    encoder_sha = None
    if train_mode == 'last-stage':
        save_file(encoder.state_dict(), str(args.output/'encoder.safetensors'))
        encoder_sha = digest(args.output/'encoder.safetensors')
    # Saved predictions keep the global split so different regional runs remain pairable.
    all_parts = split_indices(labels)
    partitions = {name: indices for name, indices in zip(('train', 'validation', 'test'), (train, val, test))}
    report = {'schema_version': SCHEMA_VERSION, 'mode': 'shadow', 'clinical_validation': False,
              'outputs': list(OUTPUTS), 'preprocess_version': PREPROCESS_VERSION, 'input_size': args.size,
              'code_sha256': provenance,
              'training_config': {'seed':17,'threads':args.threads,'device':device.type,
                                  'gpu_name':torch.cuda.get_device_name(device) if device.type == 'cuda' else None,
                                  'head_lr':.003,
                                  'tail_lr':1e-5 if train_mode=='last-stage' else None,'batch_size':batch_size,
                                  'torch':torch.__version__},
              'region_scope': region_scope, 'train_mode': train_mode, 'views': views,
              'selected_epoch': best_epoch if train_mode == 'last-stage' else args.epochs,
              'encoder_sha256': encoder_sha,
              'backbone': args.backbone, 'backbone_sha256': expected, 'loss': args.loss, 'epochs': args.epochs,
              'model_sha256': digest(args.output / 'model.ts'), 'head_sha256': digest(args.output / 'head.safetensors'),
              'labels_sha256': digest(args.labels), 'thresholds': thresholds, 'metrics': metrics,
              'split': {name: {'images': len(idx), 'studies': int(labels.study_key.iloc[idx].nunique())}
                        for name, idx in partitions.items()},
              'changed_pixel_hashes': changed_hashes, 'duration_seconds': time.perf_counter()-start,
              'validation_scope': 'fixed study-held-out internal split, known region; no external or patient-identity validation'}
    (args.output / 'history.json').write_text(json.dumps(history, indent=2, allow_nan=False)+'\n')
    (args.output / 'model.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    # Hash group identifiers in shareable reports; original paths/UIDs stay in source labels.
    split_map = {int(i): name for name, ids in zip(('train','validation','test'),all_parts) for i in ids}
    rows = [{'label_row': i, 'split': split_map[i],
             'study_hash': hashlib.sha256(str(row.study_key).encode()).hexdigest(),
             **{name+'_score': float(scores[i,j]) for j,name in enumerate(OUTPUTS)}}
            for i,row in enumerate(labels.itertuples())]
    (args.output / 'predictions.json').write_text(json.dumps(rows, indent=2)+'\n')
    print(json.dumps(report, indent=2), flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--labels', type=Path, default=Path(__file__).parent/'labels/image_labels.csv')
    parser.add_argument('--weights', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--backbone', choices=['convnextv2_tiny', 'tf_efficientnet_b4'], default='convnextv2_tiny')
    parser.add_argument('--region', choices=['all','spine','hip'], default='all')
    parser.add_argument('--train-mode', choices=['head','last-stage'], default='head')
    parser.add_argument('--views', choices=['full','full-center'], default='full')
    parser.add_argument('--loss', choices=['bce', 'asl'], default='bce')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--size', type=int, default=224)
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--device', choices=['cpu', 'cuda', 'auto'], default='cpu',
                        help='Training device; exported TorchScript remains CPU-loadable')
    run(parser.parse_args())


if __name__ == '__main__':
    main()
