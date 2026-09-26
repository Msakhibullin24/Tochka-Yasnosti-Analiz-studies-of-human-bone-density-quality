"""Paired anatomy transfer audit; unlabeled coverage is never called accuracy."""
import argparse
import csv
import hashlib
import json
import warnings
from pathlib import Path

import cv2
import numpy as np
import torch

from anatomy_training_data import load_record
from dxaqc.dicom_io import read_dxa
from dxaqc.learned_anatomy import LearnedAnatomy, decode_masks, mask_regions, numbered_axis, spatial_features
from refine_lumbar_masks import lumbar_crop
from train_anatomy_masks import metrics


def paired_stability(first, second):
    """Dice on prediction unions; empty predictions have undefined stability."""
    first, second = np.asarray(first), np.asarray(second)
    if first.shape != second.shape or first.ndim != 2:
        raise ValueError('Expected aligned label rasters')
    values = []
    for label in range(13, 17):
        a, b = first == label, second == label
        denominator = int(a.sum() + b.sum())
        if denominator:
            values.append(2*int((a & b).sum())/denominator)
    return float(np.mean(values)) if values else None


def coverage_summary(rows):
    unique = {}
    for row in rows:
        key = row['pixel_sha256']
        availability = {p: row['axis_candidates_deg'][p] is not None for p in ('original', 'inverted')}
        if key in unique and availability != unique[key]:
            raise ValueError('Duplicate pixels disagree on axis availability')
        unique[key] = availability
    return {'unique_pixel_rasters': len(unique),
            'unique_pixel_axis_available': {p: sum(a[p] for a in unique.values()) for p in ('original', 'inverted')}}


def infer(models, pixels):
    feature = torch.from_numpy(spatial_features(pixels).astype(np.float32))[None]
    with torch.inference_mode():
        return {name: model.model(feature)[0].numpy() for name, model in models.items()}


def resolve_source(root, name):
    root = Path(root).resolve()
    path = (root/name).resolve()
    if not path.is_relative_to(root):
        raise ValueError("Source path escapes dataset root")
    if not path.is_file():
        raise ValueError("Source DICOM does not exist")
    return path


def audit(base, refined, predictions, source_root, output, candidate_name="lumbar_refinement", skip_annotated=False, baseline_name="stage1"):
    if output.exists():
        raise ValueError('Choose a new output report')
    with predictions.open(encoding='utf-8-sig', newline='') as stream:
        rows = [r for r in csv.DictReader(stream) if r['anatomical_region'] == 'Поясничный отдел позвоночника']
    if not rows:
        raise ValueError('No spine source rows')
    sources = [resolve_source(source_root, row['path_to_file']) for row in rows]
    if not candidate_name or not baseline_name or candidate_name == baseline_name:
        raise ValueError('Candidate name must differ from baseline')
    torch.set_num_threads(4)
    models = {baseline_name: LearnedAnatomy(base/'anatomy_masks.pt'),
              candidate_name: LearnedAnatomy(refined/'anatomy_masks.pt')}
    inventory = [] if skip_annotated else json.loads((base/'inventory.json').read_text())['rows']
    annotated = []
    for row in ([] if skip_annotated else inventory):
        if row['split'] != 'test':
            continue
        pixels, target = load_record(row)
        views = [('original', pixels, target)]
        if row['source'] == 'aasce':
            cropped, annotation = lumbar_crop(pixels, target)
            views.append(('lumbar_crop', cropped, annotation))
        for view, raster, annotation in views:
            reference = cv2.resize(annotation.astype(np.float32), (40, 40), interpolation=cv2.INTER_NEAREST).astype(np.int64)
            for polarity, image in [('original', raster), ('inverted', 255-raster)]:
                for name, logits in infer(models, image).items():
                    entries = metrics(logits.argmax(0), reference)
                    annotated.append({'model': name, 'source': row['source'], 'group': row['group'],
                                      'view': view, 'polarity': polarity,
                                      'dice': float(np.mean([m['dice'] for m in entries]))})
    unlabelled = []
    for i, row in enumerate(rows):
        image = read_dxa(sources[i])
        variants = {p: infer(models, raster) for p, raster in [('original', image.pixels), ('inverted', 255-image.pixels)]}
        for name in models:
            rasters = {p: decode_masks(outputs[name], image.pixels.shape) for p, outputs in variants.items()}
            axes = {p: numbered_axis(mask_regions(raster), image.pixel_mm_x or image.pixel_mm, image.pixel_mm)
                    for p, raster in rasters.items()}
            unlabelled.append({'model': name, 'source_sha256': hashlib.sha256(sources[i].read_bytes()).hexdigest(),
                               'pixel_sha256': image.pixel_sha256, 'axis_candidates_deg': axes,
                               'polarity_prediction_dice': paired_stability(rasters['original'], rasters['inverted']),
                               'clinical_accuracy': None})
        if (i+1)%25 == 0:
            print(f'GE transfer {i+1}/{len(rows)}', flush=True)
    summary = {}
    for name in models:
        labelled = {}
        for source in ('aasce', 'ramathibodi'):
            for view in ('original', 'lumbar_crop'):
                for polarity in ('original', 'inverted'):
                    selected = [r['dice'] for r in annotated if (r['model'], r['source'], r['view'], r['polarity']) == (name, source, view, polarity)]
                    if selected:
                        labelled[f'{source}/{view}/{polarity}'] = {'mean_image_dice': float(np.mean(selected)), 'images': len(selected)}
        selected = [r for r in unlabelled if r['model'] == name]
        stability = [r['polarity_prediction_dice'] for r in selected if r['polarity_prediction_dice'] is not None]
        summary[name] = {'annotated': labelled, 'unlabelled_ge': {
            **coverage_summary(selected), 'images': len(selected), 'axis_available': {p: sum(r['axis_candidates_deg'][p] is not None for r in selected) for p in ('original', 'inverted')},
            'mean_polarity_prediction_dice': float(np.mean(stability)) if stability else None,
            'clinical_accuracy': None}}
    report = {'protocol': 'Paired fixed source holdout, original/cropped field and polarity; diagnostic post-selection audit only',
              'clinical_validation': False, 'annotated_evaluation_performed': not skip_annotated, 'summary': summary, 'annotated': annotated, 'unlabelled_ge': unlabelled,
              'models_sha256': {k: v.sha256 for k, v in models.items()},
              'code_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'limitations': ['GE has no reference masks: axis availability and stability cannot establish numbering accuracy.',
                             'Published source holdouts have been inspected previously; this is not an independent confirmation.',
                             'AASCE image groups do not establish patient separation.',
                             'Inverted rasters are stress tests; DICOM decoding already normalizes polarity.',
                             'Source rows are selected by predicted region, not reference region; duplicate pixel counts are reported separately.']}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('base', 'refined', 'predictions', 'source-root', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--baseline-name', default='stage1')
    parser.add_argument('--candidate-name', default='lumbar_refinement')
    parser.add_argument('--skip-annotated', action='store_true', help='Use when source holdout was already evaluated by training report')
    args = parser.parse_args()
    warnings.filterwarnings('ignore', message='Invalid value for VR UI:.*')
    warnings.filterwarnings('ignore', message='TypedStorage is deprecated.*')
    audit(args.base, args.refined, args.predictions, args.source_root, args.output, args.candidate_name, args.skip_annotated, args.baseline_name)
