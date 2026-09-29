"""DXA-only folder-group-held-out localization of the vendor neck bone ROI.

This tests a vendor-defined measurement area, not anatomical neck contours or
clinical QC. No ordinary radiographs or organizer quality labels are used.
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

from dxaqc.dicom_io import read_dxa
from dxaqc.embedding import WEIGHTS_SHA256
from dxaqc.learned_anatomy import mask_head, spatial_features
from evaluate_external_roi_masks import read_seg_nrrd, sha256


def dice(predicted: np.ndarray, truth: np.ndarray) -> float:
    predicted, truth = np.asarray(predicted, bool), np.asarray(truth, bool)
    if predicted.shape != truth.shape:
        raise ValueError('Mask sizes differ')
    total = int(predicted.sum() + truth.sum())
    return 2 * int((predicted & truth).sum()) / total if total else 1.


def load_cases(root: Path):
    rows, features, targets = [], [], []
    seen = set()
    for patient in sorted((root/'Annotation').iterdir()):
        if not patient.is_dir():
            continue
        image_path = patient/'images/hip_image.dcm'
        mask_path = patient/'segmentations/hip_image.seg.nrrd'
        image = read_dxa(image_path)
        reference = read_seg_nrrd(mask_path, image.pixels.shape)['Femoral_Neck_bone_area']
        if not reference.any() or image.pixel_sha256 in seen:
            raise ValueError('Empty ROI or duplicate DXA pixels')
        seen.add(image.pixel_sha256)
        features.append(spatial_features(image.pixels))
        targets.append(cv2.resize(reference.astype(np.uint8), (40, 40), interpolation=cv2.INTER_NEAREST))
        rows.append({'group_sha256': hashlib.sha256(patient.name.encode()).hexdigest(),
                     'image_sha256': sha256(image_path), 'mask_sha256': sha256(mask_path),
                     'pixel_sha256': image.pixel_sha256,
                     'label_basis': 'publisher_femoral_neck_bone_area_ROI',
                     'native_shape': list(image.pixels.shape)})
    if len(rows) < 10 or len({row['group_sha256'] for row in rows}) != len(rows):
        raise ValueError('Ten independent annotated DXA folders required')
    return rows, np.asarray(features), np.asarray(targets, dtype=np.float32)


def fit_head(features: np.ndarray, targets: np.ndarray, train: np.ndarray, seed: int):
    torch.manual_seed(seed)
    model = mask_head(1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.0001)
    x = torch.from_numpy(features[train].astype(np.float32))
    y = torch.from_numpy(targets[train, None].astype(np.float32))
    positives = float(y.sum())
    negatives = float(y.numel() - positives)
    if positives <= 0:
        raise ValueError('No positive training ROI pixels')
    pos_weight = torch.tensor(min(negatives / positives, 20.))
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


def run(root: Path, output: Path):
    if output.exists():
        raise ValueError('Choose a new experiment directory')
    torch.set_num_threads(4)
    rows, features, targets = load_cases(root)
    folds = KFold(n_splits=5, shuffle=True, random_state=17)
    cases = []
    for fold, (train, test) in enumerate(folds.split(rows)):
        assert not {rows[i]['group_sha256'] for i in train} & {rows[i]['group_sha256'] for i in test}
        model = fit_head(features, targets, train, seed=17 + fold)
        prior = targets[train].mean(axis=0) >= .5
        model.eval()
        with torch.inference_mode():
            x = torch.from_numpy(features[test].astype(np.float32))
            predictions = model(x).sigmoid().numpy()[:, 0] >= .5
        for index, predicted in zip(test, predictions):
            truth = targets[index] > .5
            cases.append({**rows[index], 'fold': fold,
                          'prediction_dice_40': dice(predicted, truth),
                          'layout_prior_dice_40': dice(prior, truth),
                          'prediction_pixels_40': int(predicted.sum()),
                          'reference_pixels_40': int(truth.sum())})
    report = {'scope': 'DXA-only research; released QC unchanged',
              'source_url': 'https://data.mait.ai.in.th/dataset/bone-mineral-density-and-vertebral-fracture-asseessment-public-sharing',
              'code_sha256': sha256(Path(__file__)),
              'encoder_sha256': WEIGHTS_SHA256,
              'protocol': 'Five fixed folds over 10 publisher Annotation case folders; 8 train, 2 held out per fold; fixed 25 epochs; no test tuning',
              'images': len(cases), 'groups': len({c['group_sha256'] for c in cases}),
              'mean_prediction_dice_40': float(np.mean([c['prediction_dice_40'] for c in cases])),
              'mean_layout_prior_dice_40': float(np.mean([c['layout_prior_dice_40'] for c in cases])),
              'clinical_validation': False, 'model_affects_decision': False,
              'limitations': ['Reference is vendor bone-area measurement ROI, not an independent anatomical neck contour.',
                              'Publisher case folders are split identities; patient identity across folders is not independently verified.',
                              'Only 10 Hologic images with ROI masks; no GE anatomy masks.',
                              '40x40 Dice cannot verify neck axis, trochanters, rotation, soft tissue or QC correctness.',
                              'No model trained here is included in the release.'],
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
