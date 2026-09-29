"""Research-only paired DXA hip positioning probe with frozen BiomedCLIP features.

The organizer label combines positioning and rotation. It cannot verify
rotation subtypes, femoral-neck landmarks, or clinical deployment suitability.
"""
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

from dxaqc.dicom_io import read_any
from dxa_hip_neck_roi_cv import sha256
from dxa_hip_vit_qc_cv import evaluate
from hip_rotation_ablation import run as baseline_run
from source_integrity import inspect_sources
from train import build_table

HERE = Path(__file__).resolve().parents[1]
MODEL_REPO = 'microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224'
MODEL_REVISION = '9f341de24bfb00180f1b847274256e9b65a3a32e'
MODEL_SHA256 = '52cc993c5c5ff962bd0c60931874bc001e7e9b41666a385530f4a036294576be'
MODEL_NAME = 'biomedclip_local_dxa_probe'


def vision_features(dataset: Path, labels: pd.DataFrame, model_dir: Path,
                    audited_hashes: dict[str, str]):
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
    selected = np.flatnonzero(labels.region.isin(['hip_left', 'hip_right']).to_numpy())
    result = []
    for serial, index in enumerate(selected, 1):
        row = labels.iloc[index]
        image = read_any(dataset / row.first_source_path, getattr(row, 'pixel_mm', None))
        if image.pixel_sha256 != audited_hashes[row.first_source_path]:
            raise ValueError('Organizer DXA pixel hash changed since source audit')
        if image.pixels.shape != (int(row.rows), int(row.columns)):
            raise ValueError('Organizer DXA geometry changed since source audit')
        pixels = image.pixels[:, ::-1] if row.region == 'hip_left' else image.pixels
        tensor = preprocess(Image.fromarray(np.uint8(pixels)).convert('RGB'))[None]
        with torch.inference_mode():
            feature = model.encode_image(tensor)[0]
        result.append(feature.numpy())
        if serial % 25 == 0:
            print(f'Encoded {serial}/{len(selected)} hip DXA', flush=True)
    features = np.full((len(labels), len(result[0])), np.nan, dtype=np.float32)
    features[selected] = np.asarray(result, dtype=np.float32)
    if len(selected) != 150 or not np.isfinite(features[selected]).all():
        raise ValueError('Incomplete organizer hip features')
    return selected, features


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
    variants['biomedclip_image_lr'] = variants.pop('vit_cls_lr')
    table = table.rename(columns={
        'vit_cls_lr_score': 'biomedclip_image_lr_score',
        'vit_cls_lr_pred': 'biomedclip_image_lr_pred',
    })
    report = {
        'scope': 'Research-only organizer DXA hip positioning/rotation label',
        'protocol': 'Same fixed 5-fold study-held-out outer CV and 4-fold inner threshold selection as hip_rotation_ablation; frozen BiomedCLIP image embedding; fixed balanced C=.01 logistic probe; no test-fold tuning',
        'images': len(table), 'studies': int(table.study.nunique()),
        'positives': int(table.target.sum()),
        'selected_pixel_sha256_current': [audited_hashes[labels.first_source_path.iloc[i]]
                                          for i in selected],
        'labels_sha256': sha256(labels_path), 'source_fingerprint': fingerprint,
        'model_url': 'https://huggingface.co/' + MODEL_REPO,
        'model_revision': MODEL_REVISION,
        'model_weight_sha256': sha256(args.model_dir / 'open_clip_pytorch_model.bin'),
        'code_sha256': sha256(Path(__file__)),
        'baseline_code_sha256': sha256(Path(baseline_run.__code__.co_filename)),
        'variants': variants,
        'baseline_reproduction': baseline_report['variants']['current_rf_lr_cnn'],
        'clinical_validation': False, 'model_affects_decision': False,
        'limitations': [
            'Combined label cannot distinguish positioning from rotation or its subtypes.',
            'Organiser data has been inspected before; exploratory comparison, not independent validation.',
            'Known region is supplied and patient identity across studies is unverified.',
            'No independent hip anatomy landmarks or projection labels.',
            'Publisher model card limits intended use to research and puts deployed use out of scope.',
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    table.to_csv(args.output.with_name(args.output.stem + '_oof.csv'), index=False)
    print(json.dumps(variants, indent=2))


if __name__ == '__main__':
    main()
