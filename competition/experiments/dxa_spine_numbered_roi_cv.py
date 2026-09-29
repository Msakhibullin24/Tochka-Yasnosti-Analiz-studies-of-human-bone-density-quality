"""DXA-only held-out test of named L1-L4 vendor bone-area ROI segmentation.

The reference masks are vendor measurement ROIs, not complete vertebral bodies
or disc boundaries. This experiment never changes the released QC decisions.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.nn import functional as F
from sklearn.model_selection import KFold

from anatomy_training_data import dxa_spine_target
from dxaqc.dicom_io import read_dxa
from dxaqc.embedding import WEIGHTS_SHA256
from dxaqc.learned_anatomy import mask_head, spatial_features
from evaluate_external_roi_masks import read_seg_nrrd, sha256


LEVELS = ('L1', 'L2', 'L3', 'L4')


def load_cases(root: Path):
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
                     'pixel_sha256': image.pixel_sha256,
                     'label_basis': 'publisher_L1_L4_bone_area_ROI',
                     'native_shape': list(image.pixels.shape)})
        features.append(spatial_features(image.pixels))
        targets.append(compact)
    if len(rows) < 10 or len({row['group_sha256'] for row in rows}) != len(rows):
        raise ValueError('Ten independent annotated case folders required')
    return rows, np.asarray(features), np.asarray(targets, dtype=np.int64)


def fit_head(features, targets, training, seed):
    torch.manual_seed(seed)
    model = mask_head(5)
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


def dice(prediction, truth):
    total = int(prediction.sum() + truth.sum())
    return 2 * int((prediction & truth).sum()) / total if total else 1.


def run(root: Path, output: Path):
    if output.exists():
        raise ValueError('Choose a new experiment directory')
    torch.set_num_threads(4)
    rows, features, targets = load_cases(root)
    cases = []
    folds = KFold(n_splits=5, shuffle=True, random_state=17)
    for fold, (training, testing) in enumerate(folds.split(rows)):
        assert not {rows[i]['group_sha256'] for i in training} & {rows[i]['group_sha256'] for i in testing}
        model = fit_head(features, targets, training, 17 + fold)
        # The pixelwise modal class is learned only from the training cases.
        vote = np.stack([(targets[training] == value).sum(axis=0) for value in range(5)])
        prior = vote.argmax(axis=0)
        model.eval()
        with torch.inference_mode():
            x = torch.from_numpy(features[testing].astype(np.float32))
            predictions = model(x).argmax(dim=1).numpy()
        for index, prediction in zip(testing, predictions):
            scores = {}
            for value, name in enumerate(LEVELS, 1):
                truth = targets[index] == value
                scores[name] = {'prediction_dice_40': dice(prediction == value, truth),
                                'layout_prior_dice_40': dice(prior == value, truth),
                                'prediction_pixels_40': int((prediction == value).sum()),
                                'reference_pixels_40': int(truth.sum())}
            cases.append({**rows[index], 'fold': fold, 'levels': scores})
    prediction_scores = [case['levels'][name]['prediction_dice_40'] for case in cases for name in LEVELS]
    prior_scores = [case['levels'][name]['layout_prior_dice_40'] for case in cases for name in LEVELS]
    report = {'scope': 'DXA-only research; released QC unchanged',
              'source_url': 'https://data.mait.ai.in.th/dataset/bone-mineral-density-and-vertebral-fracture-asseessment-public-sharing',
              'code_sha256': sha256(Path(__file__)), 'encoder_sha256': WEIGHTS_SHA256,
              'protocol': 'Five fixed folds over 10 publisher Annotation case folders; 8 train, 2 held out per fold; fixed 25 epochs; no test tuning',
              'images': len(cases), 'groups': len({case['group_sha256'] for case in cases}),
              'named_regions': list(LEVELS),
              'mean_prediction_dice_40': float(np.mean(prediction_scores)),
              'mean_layout_prior_dice_40': float(np.mean(prior_scores)),
              'clinical_validation': False, 'model_affects_decision': False,
              'limitations': ['Named masks are vendor bone-area measurement ROIs, not complete vertebral body contours.',
                              'No Th12, iliac crest, disc boundary or vertebral anatomical-variant labels.',
                              'Only 10 Hologic images; no GE numbered reference masks.',
                              'Publisher case folders are split identities; patient identity across folders is not independently verified.',
                              '40x40 Dice does not establish correct clinical numbering or ROI validity.'],
              'cases': cases}
    output.mkdir(parents=True)
    (output/'evaluation.json').write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n')
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True, help='Ramathibodi BMD root')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = run(args.root, args.output)
    print(json.dumps({key: result[key] for key in ('images', 'groups', 'mean_prediction_dice_40', 'mean_layout_prior_dice_40')}))
