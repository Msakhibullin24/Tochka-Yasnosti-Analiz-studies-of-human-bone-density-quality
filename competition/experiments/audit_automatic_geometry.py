"""Audit new geometric checks on identity-checked cached anatomy, without QC targets."""
from __future__ import annotations
import argparse
from collections import Counter
import csv
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from dxaqc.dicom_io import read_dxa
from gpu_research.common import digest
from experiments.mask_anatomy_features import largest_component
from experiments.automatic_anatomy_geometry import th12_visibility, iliac_crests, hip_roi_relationships, MUSCLE_CHANNELS


def references(directory):
    meta = json.loads((directory/'probe.json').read_text())
    return meta, {r['pixel_sha256']: r for r in meta['results'] if r['polarity'] == 'bone-bright'}


def load_masks(directory, ref, source, image, historical_pixel):
    path = directory/f'{historical_pixel[:16]}-bone-bright.npy'
    if (digest(path) != ref['mask_sha256'] or digest(source) != ref['source_file_sha256']
            or image.pixel_sha256 != ref['decoded_pixel_sha256']):
        raise ValueError('Source, decoded pixel or predicted mask identity differs')
    masks = np.load(path, allow_pickle=False)
    if masks.shape[1:] != image.pixels.shape:
        raise ValueError('Mask/image coordinate system differs')
    return masks.astype(bool)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs', type=Path, default=Path('data/gpu-runs/anatomy-upgrade-2026-10-15'))
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--sources', type=Path, default=ROOT/'labels/image_labels.csv')
    parser.add_argument('--predictions', type=Path, default=Path('data/gpu-final-1-11-7-audit-2026-10-15/cli/results_extended.csv'))
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists(): raise ValueError('Choose a new geometry audit directory')
    dirs = {name: args.runs/name for name in ('spine-masks', 'spine-pelvis-masks', 'hip-masks', 'hip-masks-mirrored')}
    meta, refs = {}, {}
    for name, directory in dirs.items(): meta[name], refs[name] = references(directory)
    if (any(m['labels_sha256'] != digest(args.sources) for m in meta.values()) or
            meta['hip-masks']['checkpoint_sha256'] != meta['hip-masks-mirrored']['checkpoint_sha256'] or
            meta['hip-masks']['checkpoint_sha256'] != meta['spine-pelvis-masks']['checkpoint_sha256']):
        raise ValueError('Anatomical probe source or checkpoint identity differs')
    with args.predictions.open(newline='') as stream: predictions = list(csv.DictReader(stream))
    sources = pd.read_csv(args.sources, usecols=['first_source_path', 'pixel_sha256', 'region'])
    results, counts = [], Counter()
    for row in sources.itertuples():
        main_probe = 'spine-masks' if row.region == 'spine' else 'hip-masks'
        if row.pixel_sha256 not in refs[main_probe]: continue
        path = args.dataset/row.first_source_path
        image = read_dxa(path)
        matches = [r for r in predictions if r['path_to_file'] == row.first_source_path
                   or r['path_to_file'].endswith('/'+row.first_source_path)]
        if len(matches) != 1: raise ValueError('Prediction path identity is ambiguous')
        learned = json.loads(matches[0]['learned_anatomy'])
        masks = load_masks(dirs[main_probe], refs[main_probe][row.pixel_sha256], path, image, row.pixel_sha256)
        if row.region == 'spine':
            pelvis = load_masks(dirs['spine-pelvis-masks'], refs['spine-pelvis-masks'][row.pixel_sha256], path, image, row.pixel_sha256)
            report = {'th12': th12_visibility(masks, image.pixel_mm_x, image.pixel_mm), 'iliac_crests': iliac_crests(pelvis)}
            counts['spine_images'] += 1
            counts['th12_'+report['th12'].get('outcome', 'unavailable')] += 1
        else:
            other = load_masks(dirs['hip-masks-mirrored'], refs['hip-masks-mirrored'][row.pixel_sha256], path, image, row.pixel_sha256)
            native, mirrored = masks, other
            selected = masks.copy()
            muscle_mirrored = largest_component(other[6] | other[7]).sum() > largest_component(masks[6] | masks[7]).sum()
            for a, b in ((6, 7), (8, 9)):
                if largest_component(other[a] | other[b]).sum() > largest_component(masks[a] | masks[b]).sum():
                    selected[a], selected[b] = other[a], other[b]
            selected[list(MUSCLE_CHANNELS)] = (mirrored if muscle_mirrored else native)[list(MUSCLE_CHANNELS)]
            report = hip_roi_relationships(selected, learned.get('landmark_points'), learned['neck_roi_assessment'], image.pixel_mm_x, image.pixel_mm)
            report['muscle_model_input_branch'] = 'mirrored_then_inverse_mapped' if muscle_mirrored else 'native'
            counts['hip_images'] += 1
            counts['hip_geometry_'+report['status']] += 1
            if report['status'] == 'evaluated':
                counts['roi_predicted_soft_tissue_both_ends'] += report['soft_tissue_sides']['predicted_soft_tissue_at_both_ends']
                counts['roi_estimated_gt_compartment_excluded'] += bool(report['greater_trochanter_compartment']['estimated_compartment_excluded'])
        results.append({'historical_pixel_sha256': row.pixel_sha256, 'decoded_pixel_sha256': image.pixel_sha256,
                        'region': row.region, 'geometry': report})
    args.output.mkdir(parents=True)
    audit = {'protocol': __doc__, 'counts': dict(counts), 'results': results,
             'source_manifest_sha256': digest(args.sources), 'predictions_sha256': digest(args.predictions),
             'code_sha256': digest(Path(__file__)), 'geometry_code_sha256': digest(Path(__file__).with_name('automatic_anatomy_geometry.py')),
             'checkpoint_sha256': {k: m['checkpoint_sha256'] for k, m in meta.items()},
             'qc_target_values_used': False, 'qc_predictions_used_to_choose_geometric_rules': False,
             'clinical_validation': False, 'anatomical_accuracy_evaluation': False,
             'limitations': ['Body and GT boundaries are model-derived estimates, not expert contours.',
                 'A missing muscle prediction does not prove missing soft tissue.',
                 'No numeric pelvic-overlap tolerance is invented.']}
    (args.output/'evaluation.json').write_text(json.dumps(audit, indent=2)+'\n')
    print(json.dumps(audit['counts']), flush=True)


if __name__ == '__main__': main()
