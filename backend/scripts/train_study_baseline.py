"""Exploratory CPU baseline: any quality violation in a whole study.

Uses organizer study-level labels without guessing per-image anatomy. This is
not the competition's per-image model and must not be connected to its API.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from PIL import Image, ImageOps
import torch
import torchvision
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, balanced_accuracy_score,
                             confusion_matrix, f1_score, roc_auc_score)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]


def connected_groups(records, pairs):
    """Conservatively keep even unconfirmed near-duplicate candidates together."""
    parent = {r['source_study_key']: r['source_study_key'] for r in records}
    image_study = {r['image_id']: r['source_study_key'] for r in records}

    def find(key):
        while parent[key] != key:
            parent[key] = parent[parent[key]]
            key = parent[key]
        return key

    def union(a, b):
        a, b = find(a), find(b)
        parent[max(a, b)] = min(a, b)

    by_pixels = defaultdict(list)
    for record in records:
        by_pixels[record['pixel_sha256']].append(record['source_study_key'])
    for keys in by_pixels.values():
        for key in keys[1:]:
            union(keys[0], key)
    for pair in pairs:
        union(image_study[pair['first_image']], image_study[pair['second_image']])
    return {key: find(key) for key in parent}


def study_targets(labels):
    targets = {}
    for row in labels:
        observed = [v['quality_class'] for v in row['regions'].values() if v['quality_class'] is not None]
        if observed:
            if any(v not in (0, 1) for v in observed):
                raise ValueError('Non-binary organizer label')
            targets[row['source_study_key']] = max(observed)
    return targets


def metrics(y, p):
    predicted = (p >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, predicted, labels=[0, 1]).ravel()
    return {'roc_auc': float(roc_auc_score(y, p)) if len(set(y)) == 2 else None,
            'average_precision': float(average_precision_score(y, p)) if np.any(y) else None,
            'f1': float(f1_score(y, predicted, zero_division=0)),
            'balanced_accuracy': float(balanced_accuracy_score(y, predicted)),
            'sensitivity': float(tp / (tp + fn)) if tp + fn else None,
            'specificity': float(tn / (tn + fp)) if tn + fp else None,
            'confusion_matrix_tn_fp_fn_tp': [int(v) for v in (tn, fp, fn, tp)]}


def learner():
    return make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=2000, random_state=17))


def run(data: Path, checkpoint: Path, output: Path, threads: int):
    if output.exists():
        raise FileExistsError('Refusing to overwrite an existing run')
    started = time.perf_counter()
    torch.set_num_threads(threads)
    torch.manual_seed(17)
    records = read_jsonl(data / 'manifest.jsonl')
    labels = read_jsonl(data / 'study_labels.jsonl')
    targets = study_targets(labels)
    groups = connected_groups(records, read_jsonl(data / 'near_duplicate_candidates.jsonl'))
    checkpoint_sha = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    if not checkpoint_sha.startswith('f37072fd'):
        raise ValueError('Expected torchvision ResNet18 IMAGENET1K_V1 checkpoint')
    encoder = torchvision.models.resnet18(weights=None)
    encoder.load_state_dict(torch.load(checkpoint, map_location='cpu', weights_only=True))
    encoder.fc = torch.nn.Identity()
    encoder.eval()
    mean = torch.tensor([0.485, 0.456, 0.406])[:, None, None]
    std = torch.tensor([0.229, 0.224, 0.225])[:, None, None]
    embeddings = []
    with torch.inference_mode():
        for offset in range(0, len(records), 8):
            tensors = []
            for record in records[offset:offset + 8]:
                path = (data / record['image_path']).resolve()
                if data.resolve() not in path.parents:
                    raise ValueError('Image outside dataset root')
                with Image.open(path) as im:
                    im = ImageOps.pad(im.convert('RGB'), (320, 320), method=Image.Resampling.BILINEAR, color=0)
                    tensor = torch.from_numpy(np.asarray(im).copy()).permute(2, 0, 1).float() / 255
                tensors.append((tensor - mean) / std)
            embeddings.extend(encoder(torch.stack(tensors)).numpy())
            print(f'Features {min(offset + 8, len(records))}/{len(records)}', flush=True)
    features_seconds = time.perf_counter() - started
    by_study = defaultdict(list)
    for index, record in enumerate(records):
        by_study[record['source_study_key']].append(index)
    keys = sorted(set(by_study) & set(targets))
    x, geometry = [], []
    for key in keys:
        indices = by_study[key]
        vectors = np.asarray([embeddings[i] for i in indices])
        x.append(np.concatenate([vectors.mean(axis=0), vectors.max(axis=0)]))
        sizes = np.asarray([[records[i]['rows'], records[i]['columns']] for i in indices])
        geometry.append(np.r_[len(indices), sizes.mean(axis=0), sizes.max(axis=0)])
    x, geometry = np.asarray(x), np.asarray(geometry)
    y = np.asarray([targets[k] for k in keys])
    g = np.asarray([groups[k] for k in keys])
    probabilities, geometry_probabilities = np.zeros(len(y)), np.zeros(len(y))
    folds, assignments = [], np.zeros(len(y), dtype=int)
    for fold, (train, test) in enumerate(StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=17).split(x, y, g)):
        assert not set(g[train]) & set(g[test])
        if len(set(y[train])) != 2:
            raise ValueError('Training fold has only one class')
        probabilities[test] = learner().fit(x[train], y[train]).predict_proba(x[test])[:, 1]
        geometry_probabilities[test] = learner().fit(geometry[train], y[train]).predict_proba(geometry[test])[:, 1]
        assignments[test] = fold
        folds.append({'fold': fold, 'train_studies': len(train), 'test_studies': len(test),
                      'test_positive': int(y[test].sum()), 'metrics': metrics(y[test], probabilities[test])})
        print(f'Fold {fold}: {folds[-1]["metrics"]}', flush=True)
    # Fixed-model, group bootstrap of OOF predictions; not independent clinical validation.
    rng = np.random.default_rng(17)
    unique_groups = np.unique(g)
    group_indices = {group: np.flatnonzero(g == group) for group in unique_groups}
    auc_samples, f1_samples = [], []
    for _ in range(1000):
        indices = np.concatenate([group_indices[k] for k in rng.choice(unique_groups, len(unique_groups), replace=True)])
        if len(set(y[indices])) == 2:
            auc_samples.append(roc_auc_score(y[indices], probabilities[indices]))
            f1_samples.append(f1_score(y[indices], probabilities[indices] >= 0.5, zero_division=0))
    full_model = learner().fit(x, y)
    report = {
        'task': 'auxiliary_study_level_any_violation_NOT_per_image_competition_model',
        'status': 'exploratory_baseline', 'studies': len(keys), 'positive_studies': int(y.sum()),
        'images': len(records), 'conservative_groups': len(unique_groups),
        'largest_group_studies': max(int(np.sum(g == k)) for k in unique_groups),
        'grouping': 'study_folder + exact_pixels + all_unconfirmed_dhash_candidates',
        'encoder': 'resnet18_imagenet1k_v1_frozen', 'checkpoint_sha256': checkpoint_sha,
        'input': '320x320 letterbox; no crop or augmentation',
        'aggregation': 'concatenated_mean_and_max_embeddings', 'classifier': 'StandardScaler + LogisticRegression(C=0.1)',
        'validation': '5-fold stratified group CV; fixed hyperparameters; fixed threshold 0.5',
        'dataset_version': json.loads((data / 'summary.json').read_text())['dataset_version'],
        'label_sha256': hashlib.sha256((data / 'study_labels.jsonl').read_bytes()).hexdigest(),
        'metrics': metrics(y, probabilities), 'geometry_only_control': metrics(y, geometry_probabilities),
        'constant_score_control': metrics(y, np.full(len(y), 0.5)),
        'bootstrap_95_ci': {'roc_auc': np.percentile(auc_samples, [2.5, 97.5]).tolist(),
                            'f1': np.percentile(f1_samples, [2.5, 97.5]).tolist()},
        'folds': folds, 'feature_extraction_seconds': round(features_seconds, 3),
        'total_seconds': round(time.perf_counter() - started, 3), 'cpu_threads': threads,
        'limitations': ['No image-level anatomy mapping or defect localization',
            'Original overall labels preserved, including unresolved criterion disagreements',
            'Study label is OR over observed regions; missing region labels remain unknown',
            'Patient independence cannot be verified',
            'Unconfirmed near-duplicate candidates conservatively merged, not diagnosed as duplicates',
            'Study image count and anatomy composition can confound predictions',
            'Bootstrap describes this exploratory CV, not independent competition or clinical accuracy'],
    }
    output.mkdir(parents=True)
    (output / 'metrics.json').write_text(json.dumps(report, indent=2) + '\n')
    with (output / 'oof_predictions.jsonl').open('w') as stream:
        for i, key in enumerate(keys):
            stream.write(json.dumps({'study_key': key, 'group': groups[key], 'fold': int(assignments[i]),
                'label': int(y[i]), 'probability': float(probabilities[i]),
                'geometry_probability': float(geometry_probabilities[i])}) + '\n')
    scaler, classifier = full_model.steps[0][1], full_model.steps[1][1]
    np.savez(output / 'linear_model.npz', mean=scaler.mean_, scale=scaler.scale_,
             coef=classifier.coef_, intercept=classifier.intercept_, classes=classifier.classes_)
    np.savez(output / 'features.npz', x=x, y=y, study_keys=np.asarray(keys), groups=g, folds=assignments)
    (output / 'README.txt').write_text('Auxiliary study-level experiment only. Not a per-image DXA QC model.\nSee metrics.json for limitations. Source identifiers retained locally; do not publish this run as patient data.\n')
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--threads', type=int, default=4)
    args = parser.parse_args()
    if args.threads < 1:
        parser.error('--threads must be positive')
    run(args.data, args.checkpoint, args.output, args.threads)
