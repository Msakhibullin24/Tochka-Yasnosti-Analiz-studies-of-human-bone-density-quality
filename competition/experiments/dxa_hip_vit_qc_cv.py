"""Study-held-out DXA hip positioning probe with frozen DINOv2 CLS features.

The organiser label combines positioning and rotation. This exploratory test
cannot measure lesser-trochanter rotation subtypes or anatomical landmarks.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from transformers import Dinov2Model

from dxaqc.dicom_io import read_any
from dxaqc.model import SEED, best_f1_threshold
from dxa_hip_neck_roi_cv import sha256
from hip_rotation_ablation import f1_counts, run as baseline_run
from source_integrity import inspect_sources
from train import build_table

HERE = Path(__file__).resolve().parents[1]
MODEL_REPO = 'facebook/dinov2-base'
MODEL_REVISION = 'f9e44c814b77203eaa57a6bdbbd535f21ede1415'
MODEL_SHA256 = 'd73036b56966966d07975d696bde331762f37297e2f095de8cea0040c3aa0841'


def vision_features(dataset: Path, labels: pd.DataFrame, model_dir: Path,
                    audited_hashes: dict[str, str]):
    if sha256(model_dir / 'model.safetensors') != MODEL_SHA256:
        raise ValueError('DINOv2 checkpoint checksum mismatch')
    source = json.loads((model_dir / 'source.json').read_text())
    if source['repo'] != MODEL_REPO or source['revision'] != MODEL_REVISION:
        raise ValueError('DINOv2 checkpoint provenance mismatch')
    processor = json.loads((model_dir / 'preprocessor_config.json').read_text())
    mean = torch.tensor(processor['image_mean'], dtype=torch.float32)[None, :, None, None]
    std = torch.tensor(processor['image_std'], dtype=torch.float32)[None, :, None, None]
    if mean.shape != (1, 3, 1, 1) or std.shape != mean.shape or (std <= 0).any():
        raise ValueError('Invalid DINOv2 normalization')
    torch.set_num_threads(4)
    encoder = Dinov2Model.from_pretrained(model_dir, local_files_only=True).eval()
    if encoder.config.hidden_size != 768 or encoder.config.patch_size != 14:
        raise ValueError('Expected DINOv2 base patch-14')
    selected = np.flatnonzero(labels.region.isin(['hip_left', 'hip_right']).to_numpy())
    result = np.full((len(labels), 768), np.nan, dtype=np.float32)
    for serial, index in enumerate(selected, 1):
        row = labels.iloc[index]
        image = read_any(dataset / row.first_source_path, getattr(row, 'pixel_mm', None))
        if image.pixel_sha256 != audited_hashes[row.first_source_path]:
            raise ValueError('Organizer DXA pixel hash changed since source audit')
        if image.pixels.shape != (int(row.rows), int(row.columns)):
            raise ValueError('Organizer DXA geometry changed since source audit')
        pixels = image.pixels[:, ::-1] if row.region == 'hip_left' else image.pixels
        pixels = cv2.resize(pixels, (518, 518), interpolation=cv2.INTER_AREA)
        tensor = torch.from_numpy(pixels.astype(np.float32) / 255)[None, None].repeat(1, 3, 1, 1)
        with torch.inference_mode():
            cls = encoder(pixel_values=(tensor - mean) / std).last_hidden_state[0, 0]
        result[index] = cls.numpy()
        if serial % 25 == 0:
            print(f'Encoded {serial}/{len(selected)} hip DXA', flush=True)
    if len(selected) != 150 or not np.isfinite(result[selected]).all():
        raise ValueError('Incomplete organizer hip features')
    return selected, result


def fit_probe(x, y, train, test):
    model = make_pipeline(SimpleImputer(strategy='median'), StandardScaler(),
                          LogisticRegression(C=.01, class_weight='balanced', max_iter=5000,
                                             random_state=SEED))
    model.fit(x[train], y[train])
    return model.predict_proba(x[test])[:, 1]


def evaluate(labels, vit_features, baseline_table):
    selected = np.flatnonzero(labels.region.isin(['hip_left', 'hip_right']).to_numpy())
    y = labels.loc[selected, 'hip_position_rotation'].to_numpy(dtype=int)
    groups = labels.loc[selected, 'study_key'].to_numpy()
    if len(selected) != 150 or int(y.sum()) != 36:
        raise ValueError('Unexpected organizer DXA hip cohort')
    x = vit_features[selected]
    rows = []
    outer = StratifiedGroupKFold(5, shuffle=True, random_state=SEED)
    for fold, (tr, te) in enumerate(outer.split(selected, y, groups)):
        if set(groups[tr]) & set(groups[te]):
            raise ValueError('Outer study leakage')
        inner = StratifiedGroupKFold(4, shuffle=True, random_state=SEED + 100)
        inner_scores = np.full(len(tr), np.nan)
        for itr, iva in inner.split(tr, y[tr], groups[tr]):
            if set(groups[tr[itr]]) & set(groups[tr[iva]]):
                raise ValueError('Inner study leakage')
            inner_scores[iva] = fit_probe(x, y, tr[itr], tr[iva])
        if not np.isfinite(inner_scores).all():
            raise ValueError('Missing inner scores')
        threshold = best_f1_threshold(y[tr], inner_scores)
        scores = fit_probe(x, y, tr, te)
        for local, score in zip(te, scores):
            rows.append({'index': int(selected[local]), 'fold': fold,
                         'study': str(groups[local]), 'target': int(y[local]),
                         'vit_cls_lr_score': float(score),
                         'vit_cls_lr_pred': int(score >= threshold)})
    table = pd.DataFrame(rows).sort_values('index').reset_index(drop=True)
    baseline = baseline_table.sort_values('index').reset_index(drop=True)
    if not table[['index', 'fold', 'study', 'target']].equals(
            baseline[['index', 'fold', 'study', 'target']]):
        raise ValueError('Candidate and baseline OOF folds differ')
    table['current_score'] = baseline['current_rf_lr_cnn_score']
    table['current_pred'] = baseline['current_rf_lr_cnn_pred']
    truth = table.target.to_numpy()
    variants = {}
    for name in ('current', 'vit_cls_lr'):
        score = table[name + '_score'].to_numpy()
        prediction = table[name + '_pred'].to_numpy()
        variants[name] = {**f1_counts(truth, prediction),
                          'roc_auc': float(roc_auc_score(truth, score)),
                          'average_precision': float(average_precision_score(truth, score))}
    rng = np.random.default_rng(SEED)
    study_indices = [np.flatnonzero(table.study.to_numpy() == study) for study in table.study.unique()]
    differences = []
    for _ in range(2000):
        sampled = np.concatenate([study_indices[i] for i in rng.integers(len(study_indices), size=len(study_indices))])
        differences.append(f1_counts(truth[sampled], table.vit_cls_lr_pred.to_numpy()[sampled])['f1'] -
                           f1_counts(truth[sampled], table.current_pred.to_numpy()[sampled])['f1'])
    variants['vit_cls_lr']['paired_delta_f1_vs_current'] = variants['vit_cls_lr']['f1'] - variants['current']['f1']
    variants['vit_cls_lr']['paired_delta_f1_ci95'] = [float(z) for z in np.percentile(differences, [2.5, 97.5])]
    return variants, table


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--model-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('Choose a new report path')
    labels_path = HERE / 'labels/image_labels.csv'
    labels = pd.read_csv(labels_path)
    labels = labels[labels.quality_class.notna()].reset_index(drop=True)
    audit = inspect_sources([('organizer', labels_path, args.dataset)])
    fingerprint = hashlib.sha256(json.dumps(
        [(item['path'], item['pixel_sha256_current']) for item in audit['entries']],
        ensure_ascii=False).encode()).hexdigest()
    geometry, embeddings, _ = build_table(args.dataset, labels, HERE / '.cache', fingerprint)
    baseline_report, baseline_table = baseline_run(labels, geometry, embeddings)
    audited_hashes = {item['path']: item['pixel_sha256_current'] for item in audit['entries']}
    selected, features = vision_features(args.dataset, labels, args.model_dir, audited_hashes)
    variants, table = evaluate(labels, features, baseline_table)
    report = {'scope': 'Research-only organizer DXA hip positioning/rotation label',
              'protocol': 'Same fixed 5-fold study-held-out outer CV and 4-fold inner threshold selection as hip_rotation_ablation; frozen DINOv2 CLS embedding; fixed balanced C=.01 logistic probe; no test-fold tuning',
              'images': len(table), 'studies': int(table.study.nunique()),
              'positives': int(table.target.sum()),
              'selected_pixel_sha256_current': [audited_hashes[labels.first_source_path.iloc[i]]
                                                for i in selected],
              'labels_sha256': sha256(labels_path), 'source_fingerprint': fingerprint,
              'model_url': 'https://huggingface.co/' + MODEL_REPO,
              'model_revision': MODEL_REVISION, 'model_weight_sha256': MODEL_SHA256,
              'code_sha256': sha256(Path(__file__)),
              'baseline_code_sha256': sha256(Path(baseline_run.__code__.co_filename)),
              'variants': variants, 'baseline_reproduction': baseline_report['variants']['current_rf_lr_cnn'],
              'clinical_validation': False, 'model_affects_decision': False,
              'limitations': ['Combined label cannot distinguish positioning from rotation or its subtypes.',
                              'Organiser data has been inspected before; exploratory comparison, not independent validation.',
                              'Known region is supplied and patient identity across studies is unverified.',
                              'No independent hip anatomy landmarks or projection labels.']}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    table.to_csv(args.output.with_name(args.output.stem + '_oof.csv'), index=False)
    print(json.dumps(variants, indent=2))


if __name__ == '__main__':
    main()
