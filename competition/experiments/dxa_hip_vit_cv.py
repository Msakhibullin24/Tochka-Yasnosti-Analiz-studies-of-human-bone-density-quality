"""Held-out DXA hip vendor neck-ROI probe with frozen DINOv2 patch features.

Research only: vendor measurement masks are not anatomical neck contours.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from sklearn.model_selection import KFold
from torch import nn
from torch.nn import functional as F
from transformers import Dinov2Model

from dxa_hip_neck_roi_cv import dice
from dxa_spine_vit_cv import features_for
from dxaqc.dicom_io import read_dxa
from evaluate_external_roi_masks import read_seg_nrrd, sha256


def load_cases(root, encoder, mean, std):
    rows, features, targets, seen = [], [], [], set()
    for folder in sorted((root / 'Annotation').iterdir()):
        if not folder.is_dir():
            continue
        image_path = folder / 'images/hip_image.dcm'
        mask_path = folder / 'segmentations/hip_image.seg.nrrd'
        image = read_dxa(image_path)
        if image.pixel_sha256 in seen:
            raise ValueError('Duplicate source DXA pixels')
        seen.add(image.pixel_sha256)
        reference = read_seg_nrrd(mask_path, image.pixels.shape)['Femoral_Neck_bone_area']
        target = cv2.resize(reference.astype(np.uint8), (40, 40), interpolation=cv2.INTER_NEAREST)
        if not target.any():
            raise ValueError('Empty femoral neck ROI')
        rows.append({'group_sha256': hashlib.sha256(folder.name.encode()).hexdigest(),
                     'image_sha256': sha256(image_path), 'mask_sha256': sha256(mask_path),
                     'pixel_sha256': image.pixel_sha256,
                     'label_basis': 'publisher_femoral_neck_bone_area_ROI'})
        features.append(features_for(image.pixels, encoder, mean, std))
        targets.append(target)
    if len(rows) != 10 or len({row['group_sha256'] for row in rows}) != 10:
        raise ValueError('Exactly 10 independent publisher folders required')
    return rows, np.asarray(features), np.asarray(targets, dtype=np.float32)


def fit_head(features, targets, train, seed):
    torch.manual_seed(seed)
    model = nn.Sequential(nn.Conv2d(768, 32, 1), nn.ReLU(),
                          nn.Conv2d(32, 32, 3, padding=1), nn.ReLU(),
                          nn.Conv2d(32, 1, 1))
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.0001)
    x = torch.from_numpy(features[train].astype(np.float32))
    y = torch.from_numpy(targets[train, None].copy())
    positives = float(y.sum())
    if positives <= 0:
        raise ValueError('No positive training ROI pixels')
    pos_weight = torch.tensor(min(float(y.numel() - positives) / positives, 20.))
    rng = np.random.default_rng(seed)
    for _ in range(25):
        model.train()
        for indices in np.array_split(rng.permutation(len(train)), 2):
            logits = model(x[indices])
            truth = y[indices]
            bce = F.binary_cross_entropy_with_logits(logits, truth, pos_weight=pos_weight)
            probability = logits.sigmoid()
            soft_dice = 1 - (2 * (probability * truth).sum() + 1) / (probability.sum() + truth.sum() + 1)
            loss = bce + soft_dice
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    return model


def run(root, model_dir, model_repo, model_revision, expected_sha256, output):
    if output.exists():
        raise ValueError('Choose a new experiment directory')
    if sha256(model_dir / 'model.safetensors') != expected_sha256:
        raise ValueError('Vision transformer weight checksum mismatch')
    source = json.loads((model_dir / 'source.json').read_text())
    if source['repo'] != model_repo or source['revision'] != model_revision:
        raise ValueError('Model provenance mismatch')
    processor = json.loads((model_dir / 'preprocessor_config.json').read_text())
    mean = torch.tensor(processor['image_mean'], dtype=torch.float32)[None, :, None, None]
    std = torch.tensor(processor['image_std'], dtype=torch.float32)[None, :, None, None]
    if mean.shape != (1, 3, 1, 1) or std.shape != mean.shape or (std <= 0).any():
        raise ValueError('Invalid model normalization')
    torch.set_num_threads(4)
    encoder = Dinov2Model.from_pretrained(model_dir, local_files_only=True).eval()
    if encoder.config.hidden_size != 768 or encoder.config.patch_size != 14:
        raise ValueError('Expected DINOv2 base patch-14 architecture')
    rows, features, targets = load_cases(root, encoder, mean, std)
    del encoder
    cases = []
    for fold, (train, test) in enumerate(KFold(n_splits=5, shuffle=True, random_state=17).split(rows)):
        assert not {rows[i]['group_sha256'] for i in train} & {rows[i]['group_sha256'] for i in test}
        model = fit_head(features, targets, train, 17 + fold)
        prior = targets[train].mean(axis=0) >= .5
        model.eval()
        with torch.inference_mode():
            prediction = model(torch.from_numpy(features[test].astype(np.float32))).sigmoid().numpy()[:, 0] >= .5
        for index, proposed in zip(test, prediction):
            truth = targets[index] > .5
            cases.append({**rows[index], 'fold': fold,
                          'prediction_dice_40': dice(proposed, truth),
                          'layout_prior_dice_40': dice(prior, truth),
                          'prediction_pixels_40': int(proposed.sum()),
                          'reference_pixels_40': int(truth.sum())})
    report = {'scope': 'DXA-only research; released QC unchanged',
              'source_url': 'https://data.mait.ai.in.th/dataset/bone-mineral-density-and-vertebral-fracture-asseessment-public-sharing',
              'model_url': 'https://huggingface.co/' + model_repo,
              'model_revision': model_revision, 'model_weight_sha256': expected_sha256,
              'code_sha256': sha256(Path(__file__)),
              'feature_code_sha256': sha256(Path(features_for.__code__.co_filename)),
              'protocol': 'Same five publisher-folder folds as ResNet18 hip study; 8 train/2 held out; frozen 518-pixel DINOv2 patch features, 40x40 map, model normalization, same 25-epoch binary head protocol; no test tuning',
              'images': len(cases), 'groups': len({c['group_sha256'] for c in cases}),
              'mean_prediction_dice_40': float(np.mean([c['prediction_dice_40'] for c in cases])),
              'mean_layout_prior_dice_40': float(np.mean([c['layout_prior_dice_40'] for c in cases])),
              'clinical_validation': False, 'model_affects_decision': False,
              'limitations': ['Only 10 Hologic BMD hip images; publisher folder identity is not verified patient identity.',
                              'Reference is vendor femoral-neck bone-area measurement ROI, not anatomical neck contour or axis.',
                              'No trochanter, rotation subtype, soft tissue margin, GE domain or quality labels.',
                              'No model trained here is included in the release.'],
              'cases': cases}
    output.mkdir(parents=True)
    (output / 'evaluation.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--model-dir', type=Path, required=True)
    parser.add_argument('--model-repo', required=True)
    parser.add_argument('--model-revision', required=True)
    parser.add_argument('--expected-sha256', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = run(args.root, args.model_dir, args.model_repo, args.model_revision,
                 args.expected_sha256, args.output)
    print(json.dumps({key: result[key] for key in ('images', 'groups', 'mean_prediction_dice_40', 'mean_layout_prior_dice_40')}))
