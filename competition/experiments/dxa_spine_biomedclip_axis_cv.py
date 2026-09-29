"""Research-only spine-axis DXA probe with frozen BiomedCLIP image features."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from open_clip import create_model_and_transforms
from open_clip.factory import _MODEL_CONFIGS
from sklearn.model_selection import StratifiedGroupKFold

from dxaqc.dicom_io import read_any
from dxaqc.model import SEED, best_f1_threshold
from dxa_hip_biomedclip_qc_cv import MODEL_REPO, MODEL_REVISION, MODEL_SHA256, sha256
from dxa_hip_vit_qc_cv import fit_probe
from dxa_spine_axis_ablation import metrics
from source_integrity import inspect_sources

HERE = Path(__file__).resolve().parents[1]
MODEL_NAME = 'biomedclip_local_dxa_spine_probe'


def extract(dataset: Path, labels: pd.DataFrame, model_dir: Path,
            hashes: dict[str, str], selected: np.ndarray):
    if sha256(model_dir / 'open_clip_pytorch_model.bin') != MODEL_SHA256:
        raise ValueError('BiomedCLIP checkpoint checksum mismatch')
    source = json.loads((model_dir / 'source.json').read_text())
    if source['repo'] != MODEL_REPO or source['revision'] != MODEL_REVISION:
        raise ValueError('BiomedCLIP checkpoint provenance mismatch')
    cfg = json.loads((model_dir / 'open_clip_config.json').read_text())
    _MODEL_CONFIGS[MODEL_NAME] = cfg['model_cfg']
    torch.set_num_threads(4)
    model, _, preprocess = create_model_and_transforms(
        model_name=MODEL_NAME,
        pretrained=str(model_dir / 'open_clip_pytorch_model.bin'),
        **{f'image_{k}': v for k, v in cfg['preprocess_cfg'].items()},
    )
    model.eval()
    result = []
    for serial, index in enumerate(selected, 1):
        row = labels.iloc[index]
        image = read_any(dataset / row.first_source_path, getattr(row, 'pixel_mm', None))
        if image.pixel_sha256 != hashes[row.first_source_path]:
            raise ValueError('Organizer DXA pixel hash changed since source audit')
        if image.pixels.shape != (int(row.rows), int(row.columns)):
            raise ValueError('Organizer DXA geometry changed since source audit')
        tensor = preprocess(Image.fromarray(np.uint8(image.pixels)).convert('RGB'))[None]
        with torch.inference_mode():
            result.append(model.encode_image(tensor)[0].numpy())
        if serial % 25 == 0:
            print(f'Encoded {serial}/{len(selected)} spine DXA', flush=True)
    features = np.asarray(result, dtype=np.float32)
    if len(selected) != 99 or not np.isfinite(features).all():
        raise ValueError('Incomplete organizer spine features')
    return features


def evaluate(labels: pd.DataFrame, selected: np.ndarray, features: np.ndarray,
             baseline: pd.DataFrame):
    y = labels.loc[selected, 'spine_axis'].to_numpy(dtype=int)
    groups = labels.loc[selected, 'study_key'].to_numpy()
    if len(selected) != 99 or int(y.sum()) != 10 or len(set(groups)) != 99:
        raise ValueError('Unexpected organizer DXA spine-axis cohort')
    outer = StratifiedGroupKFold(5, shuffle=True, random_state=SEED)
    rows = []
    for fold, (tr, te) in enumerate(outer.split(selected, y, groups)):
        if set(groups[tr]) & set(groups[te]):
            raise ValueError('Outer study leakage')
        inner = StratifiedGroupKFold(3, shuffle=True, random_state=SEED + 100)
        inner_scores = np.full(len(tr), np.nan)
        for itr, iva in inner.split(tr, y[tr], groups[tr]):
            if set(groups[tr[itr]]) & set(groups[tr[iva]]):
                raise ValueError('Inner study leakage')
            inner_scores[iva] = fit_probe(features, y, tr[itr], tr[iva])
        if not np.isfinite(inner_scores).all():
            raise ValueError('Missing inner scores')
        threshold = best_f1_threshold(y[tr], inner_scores)
        scores = fit_probe(features, y, tr, te)
        for local, score in zip(te, scores):
            rows.append({
                'index': int(selected[local]), 'fold': fold,
                'study_sha256': hashlib.sha256(str(groups[local]).encode()).hexdigest(),
                'target': int(y[local]), 'biomedclip_image_lr_score': float(score),
                'biomedclip_image_lr_pred': int(score >= threshold),
            })
    table = pd.DataFrame(rows).sort_values('index').reset_index(drop=True)
    baseline = baseline.sort_values('index').reset_index(drop=True)
    if not table[['index', 'fold', 'study_sha256', 'target']].equals(
            baseline[['index', 'fold', 'study_sha256', 'target']]):
        raise ValueError('Candidate and baseline OOF folds differ')
    table['current_score'] = baseline['current_rf_lr_cnn_score']
    table['current_pred'] = baseline['current_rf_lr_cnn_pred']
    truth = table.target.to_numpy()
    variants = {}
    for name in ('current', 'biomedclip_image_lr'):
        variants[name] = metrics(truth, table[name + '_score'].to_numpy(),
                                 table[name + '_pred'].to_numpy())
    rng = np.random.default_rng(SEED)
    study_rows = [np.flatnonzero(table.study_sha256.to_numpy() == study)
                  for study in table.study_sha256.unique()]
    differences = []
    candidate = table.biomedclip_image_lr_pred.to_numpy()
    current = table.current_pred.to_numpy()
    def f1_at(indices: np.ndarray, prediction: np.ndarray) -> float:
        positives = truth[indices] == 1
        tp = int((positives & (prediction[indices] == 1)).sum())
        return 2 * tp / max(1, int((prediction[indices] == 1).sum()) + int(positives.sum()))
    for _ in range(2000):
        sample = np.concatenate([study_rows[i] for i in rng.integers(len(study_rows), size=len(study_rows))])
        differences.append(f1_at(sample, candidate) - f1_at(sample, current))
    variants['biomedclip_image_lr']['paired_delta_f1_vs_current'] = (
        variants['biomedclip_image_lr']['f1'] - variants['current']['f1'])
    variants['biomedclip_image_lr']['paired_delta_f1_ci95'] = [
        float(v) for v in np.percentile(differences, [2.5, 97.5])]
    return variants, table


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--model-dir', type=Path, required=True)
    parser.add_argument('--baseline-oof', type=Path, required=True)
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
    hashes = {item['path']: item['pixel_sha256_current'] for item in audit['entries']}
    selected = np.flatnonzero(labels.region.eq('spine').to_numpy())
    features = extract(args.dataset, labels, args.model_dir, hashes, selected)
    baseline = pd.read_csv(args.baseline_oof)
    variants, table = evaluate(labels, selected, features, baseline)
    report = {
        'scope': 'Research-only organizer DXA spine-axis label',
        'protocol': 'Same fixed 5-fold study-held-out outer CV and 3-fold inner threshold selection as dxa_spine_axis_ablation; frozen BiomedCLIP image embedding; fixed balanced C=.01 logistic probe; no test-fold tuning',
        'images': len(table), 'studies': int(table.study_sha256.nunique()),
        'positives': int(table.target.sum()),
        'selected_pixel_sha256_current': [hashes[labels.first_source_path.iloc[i]] for i in selected],
        'labels_sha256': sha256(labels_path), 'source_fingerprint': fingerprint,
        'baseline_oof_sha256': sha256(args.baseline_oof),
        'model_url': 'https://huggingface.co/' + MODEL_REPO,
        'model_revision': MODEL_REVISION, 'model_weight_sha256': MODEL_SHA256,
        'code_sha256': sha256(Path(__file__)),
        'variants': variants, 'clinical_validation': False, 'model_affects_decision': False,
        'limitations': [
            'Only ten positive DXA labels; confidence intervals are wide.',
            'The organizer axis label has no independently reviewed per-vertebra angles.',
            'Development cohort inspected previously; exploratory, not independent validation.',
            'Known spine region is supplied; routing and final quality are not evaluated.',
            'Publisher model card limits intended use to research and puts deployed use out of scope.',
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    table.to_csv(args.output.with_name(args.output.stem + '_oof.csv'), index=False)
    print(json.dumps(variants, indent=2))


if __name__ == '__main__':
    main()
