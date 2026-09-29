"""Held-out DXA L1-L4 vendor ROI probe with frozen DINOv2 patch features.

Research only. The source masks mark vendor measurement bone area, not complete
vertebrae or Th12. This script does not change the shipped quality workflow.
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

from anatomy_training_data import dxa_spine_target
from dxa_spine_numbered_roi_cv import LEVELS, dice
from dxaqc.dicom_io import read_dxa
from evaluate_external_roi_masks import read_seg_nrrd, sha256

SIZE = 518


def head():
    return nn.Sequential(nn.Conv2d(768, 32, 1), nn.ReLU(),
                         nn.Conv2d(32, 32, 3, padding=1), nn.ReLU(), nn.Conv2d(32, 5, 1))


def features_for(image, model, mean, std):
    pixels = cv2.resize(image, (SIZE, SIZE), interpolation=cv2.INTER_AREA)
    tensor = torch.from_numpy(pixels.astype(np.float32) / 255)
    tensor = tensor[None, None].repeat(1, 3, 1, 1)
    tensor = (tensor - mean) / std
    with torch.inference_mode():
        tokens = model(pixel_values=tensor).last_hidden_state[:, 1:]
        feature = tokens.transpose(1, 2).reshape(1, 768, 37, 37)
        feature = F.interpolate(feature, size=(40, 40), mode='bilinear', align_corners=False)
    return feature[0].numpy().astype(np.float16)


def load_cases(root, model, mean, std):
    rows, features, targets, seen = [], [], [], set()
    for folder in sorted((root/'Annotation').iterdir()):
        if not folder.is_dir():
            continue
        image_path = folder/'images/spine_image.dcm'
        mask_path = folder/'segmentations/spine_image.seg.nrrd'
        image = read_dxa(image_path)
        if image.pixel_sha256 in seen:
            raise ValueError('Duplicate source DXA pixels')
        seen.add(image.pixel_sha256)
        masks = read_seg_nrrd(mask_path, image.pixels.shape)
        target = dxa_spine_target(masks, image.pixels.shape)
        compact = np.where(target == -100, -100, np.where(target == 0, 0, target - 12))
        compact = cv2.resize(compact.astype(np.int16), (40, 40), interpolation=cv2.INTER_NEAREST)
        if set(range(1, 5)) - set(np.unique(compact)):
            raise ValueError('Named vertebral ROI vanished at evaluation resolution')
        rows.append({'group_sha256': hashlib.sha256(folder.name.encode()).hexdigest(),
                     'image_sha256': sha256(image_path), 'mask_sha256': sha256(mask_path),
                     'pixel_sha256': image.pixel_sha256})
        features.append(features_for(image.pixels, model, mean, std))
        targets.append(compact)
    if len(rows) != 10 or len({row['group_sha256'] for row in rows}) != 10:
        raise ValueError('Exactly 10 independent publisher folders required')
    return rows, np.asarray(features), np.asarray(targets, dtype=np.int64)


def fit(features, targets, training, seed):
    torch.manual_seed(seed)
    model = head()
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.0001)
    x = torch.from_numpy(features[training].astype(np.float32))
    y = torch.from_numpy(targets[training].copy())
    counts = np.bincount(targets[training][targets[training] >= 0].ravel(), minlength=5)
    if np.any(counts == 0):
        raise ValueError('Training lacks a required ROI class')
    weights = 1 / np.sqrt(counts.astype(float))
    weights = torch.tensor(weights / weights.mean(), dtype=torch.float32)
    rng = np.random.default_rng(seed)
    for _ in range(25):
        model.train()
        for indices in np.array_split(rng.permutation(len(training)), 2):
            loss = F.cross_entropy(model(x[indices]), y[indices], weight=weights, ignore_index=-100)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    return model


def run(root, model_dir, model_repo, model_revision, expected_sha256, output):
    if output.exists():
        raise ValueError('Choose a new experiment directory')
    if sha256(model_dir/'model.safetensors') != expected_sha256:
        raise ValueError('Vision transformer weight checksum mismatch')
    source = json.loads((model_dir/'source.json').read_text())
    if source['repo'] != model_repo or source['revision'] != model_revision:
        raise ValueError('Model provenance mismatch')
    processor = json.loads((model_dir/'preprocessor_config.json').read_text())
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
    for fold, (training, testing) in enumerate(KFold(n_splits=5, shuffle=True, random_state=17).split(rows)):
        assert not {rows[i]['group_sha256'] for i in training} & {rows[i]['group_sha256'] for i in testing}
        model = fit(features, targets, training, 17 + fold)
        vote = np.stack([(targets[training] == value).sum(axis=0) for value in range(5)])
        prior = vote.argmax(axis=0)
        model.eval()
        with torch.inference_mode():
            predictions = model(torch.from_numpy(features[testing].astype(np.float32))).argmax(1).numpy()
        for index, prediction in zip(testing, predictions):
            scores = {}
            for value, name in enumerate(LEVELS, 1):
                truth = targets[index] == value
                scores[name] = {'prediction_dice_40': dice(prediction == value, truth),
                                'layout_prior_dice_40': dice(prior == value, truth),
                                'prediction_pixels_40': int((prediction == value).sum()),
                                'reference_pixels_40': int(truth.sum())}
            cases.append({**rows[index], 'fold': fold, 'levels': scores})
    scores = [c['levels'][name]['prediction_dice_40'] for c in cases for name in LEVELS]
    priors = [c['levels'][name]['layout_prior_dice_40'] for c in cases for name in LEVELS]
    report = {'scope': 'DXA-only research; released QC unchanged',
              'source_url': 'https://data.mait.ai.in.th/dataset/bone-mineral-density-and-vertebral-fracture-asseessment-public-sharing',
              'model_url': 'https://huggingface.co/'+model_repo,
              'model_revision': model_revision, 'model_weight_sha256': expected_sha256,
              'code_sha256': sha256(Path(__file__)),
              'transformers_version': '4.44.2',
              'protocol': 'Same 5 fixed publisher-folder folds as ResNet18 study, 8 train/2 test; frozen DINOv2 base variant, 518 pixel square input, 40x40 feature map, model-specific normalization, same 25-epoch head protocol; no test tuning',
              'images': len(cases), 'groups': len({c['group_sha256'] for c in cases}),
              'mean_prediction_dice_40': float(np.mean(scores)),
              'mean_layout_prior_dice_40': float(np.mean(priors)),
              'clinical_validation': False, 'model_affects_decision': False,
              'limitations': ['Only 10 Hologic BMD images; publisher folders are unverified patient identities.',
                              'Publisher masks are L1-L4 vendor bone-area ROIs, not full vertebrae, discs or Th12.',
                              'No tested DINOv2 variant was pretrained on DXA.',
                              'No GE target-domain masks or independent clinical quality labels.'],
              'cases': cases}
    output.mkdir(parents=True)
    (output/'evaluation.json').write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n')
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
