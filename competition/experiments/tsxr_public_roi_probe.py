"""Frozen TSXR predictions versus existing publisher DXA bone-area ROIs.

Dice measures overlap of two different segmentation targets; it is not a claim
of full vertebral-body accuracy. No prompts, fine tuning or test threshold search.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import cv2
import numpy as np
import torch
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from dxaqc.dicom_io import read_dxa
from evaluate_external_roi_masks import read_seg_nrrd, overlap, sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('Choose a new output directory')
    torch.set_num_threads(4)
    predictor = nnUNetPredictor(use_mirroring=False, allow_tqdm=False, verbose=False,
                               device=torch.device('cuda'))
    predictor.initialize_from_trained_model_folder(str(args.model), use_folds=(0,),
                                                   checkpoint_name='checkpoint_best.pth')
    names = {name: int(value) for name, value in predictor.dataset_json['labels'].items()}
    if not predictor.label_manager.multilabel:
        raise ValueError('Expected independent binary label channels')
    args.output.mkdir(parents=True)
    results = []
    for folder in sorted((args.root / 'Annotation').iterdir()):
        if not folder.is_dir():
            continue
        image = read_dxa(folder / 'images/spine_image.dcm')
        references = read_seg_nrrd(folder / 'segmentations/spine_image.seg.nrrd', image.pixels.shape)
        sx, sy = image.pixel_mm_x, image.pixel_mm
        masks = np.asarray(predictor.predict_single_npy_array(
            image.pixels[None, None].astype(np.float32), {'spacing': (999 * max(sx, sy), sy, sx)})).squeeze()
        if masks.shape != (26, *image.pixels.shape):
            raise ValueError('Unexpected output raster')
        levels = {}
        for i in range(1, 5):
            prediction = masks[names[f'vertebrae-l{i}'] - 1].astype(bool)
            truth = references[f'Lumbar_{i}_bone_area']
            score = overlap(truth, prediction)
            # Prediction index is fixed by the published class, never matched to truth.
            pp = np.argwhere(prediction)
            tp = np.argwhere(truth)
            score['center_distance_mm'] = float(np.linalg.norm(
                (pp.mean(axis=0) - tp.mean(axis=0)) * [sy, sx])) if len(pp) and len(tp) else None
            score['dice_40'] = overlap(
                cv2.resize(truth.astype(np.uint8), (40, 40), interpolation=cv2.INTER_NEAREST),
                cv2.resize(prediction.astype(np.uint8), (40, 40), interpolation=cv2.INTER_NEAREST))['dice']
            levels[f'L{i}'] = score
        results.append({'case_sha256': hashlib.sha256(folder.name.encode()).hexdigest(),
                        'pixel_sha256': image.pixel_sha256, 'levels': levels,
                        'pixel_mm_source': image.pixel_mm_source})
    if not results:
        raise ValueError('No annotated cases')
    scores = [s for row in results for s in row['levels'].values()]
    report = {'model_source': 'https://zenodo.org/records/17052912',
              'reference_source': 'https://data.mait.ai.in.th/dataset/bone-mineral-density-and-vertebral-fracture-asseessment-public-sharing',
              'checkpoint_sha256': sha256(args.model / 'fold_0/checkpoint_best.pth'),
              'code_sha256': sha256(Path(__file__)), 'cases': results,
              'images': len(results), 'mean_dice_native': float(np.mean([s['dice'] for s in scores])),
              'mean_dice_40': float(np.mean([s['dice_40'] for s in scores])),
              'reference_target': 'Publisher named L1-L4 vendor bone-area ROIs, not complete vertebral-body contours',
              'clinical_validation': False, 'automatic_promotions': 0,
              'limitations': ['Only ten annotated case folders; identities across folders not independently verified.',
                              'Publisher pretrained-model training overlap not independently ruled out.',
                              'Center distances use DICOM calibration or reader fallback, recorded per case.',
                              'No Th12, iliac crest or disc-boundary reference.']}
    (args.output / 'evaluation.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({k: report[k] for k in ('images', 'mean_dice_native', 'mean_dice_40')}), flush=True)


if __name__ == '__main__':
    main()
