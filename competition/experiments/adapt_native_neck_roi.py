"""Leave-one-DICOM-study-out neck ROI geometry adaptation on published DXA masks.

A four-parameter median template in a predicted anatomical frame. Each held
study's vendor ROI is excluded from fitting. No oracle test bounding box/angle
is used to place a prediction. Ten previously inspected folders, not a new test.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import cv2
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from dxaqc.dicom_io import read_dxa
from evaluate_external_roi_masks import read_seg_nrrd, overlap, sha256
from experiments.hip_mask_features import hip_measurements
from experiments.train_silhouette_heatmaps import Heatmaps
from experiments.transfer_camera_projection import selected_masks
from experiments.neck_roi_from_mask import neck_roi


def anatomical_frame(head, neck, sx, sy):
    scale = np.array([sx, sy])
    center = np.asarray(neck) * scale
    delta = (np.asarray(head) - neck) * scale
    length = np.linalg.norm(delta)
    if length < 5 or not np.isfinite(length):
        raise ValueError('Degenerate predicted neck axis')
    axis = delta / length
    return center, np.stack([axis, [-axis[1], axis[0]]]), length


def encode_reference(mask, center, frame, length, sx, sy):
    yy, xx = np.where(mask)
    values = (np.column_stack([xx * sx, yy * sy]) - center) @ frame.T / length
    lower, upper = values.min(0), values.max(0)
    return np.r_[(lower + upper) / 2, upper - lower]


def proposal_from_template(template, center, frame, length, sx, sy, shape):
    template = np.asarray(template)
    if template.shape != (4,) or not np.isfinite(template).all() or np.any(template[2:] <= 0):
        raise ValueError('Finite positive anatomical-frame template required')
    offsets = np.array([[-.5, -.5], [.5, -.5], [.5, .5], [-.5, .5]]) * template[2:] + template[:2]
    polygon = (offsets * length @ frame + center) / [sx, sy]
    if np.any(polygon < 0) or np.any(polygon >= shape[::-1]):
        return {'status': 'unavailable', 'reason': 'proposal_outside_source_frame'}
    return {'status': 'proposed', 'purpose': 'femoral_neck', 'polygon': polygon.tolist(),
            'method': 'native_dxa_median_in_predicted_neck_frame', 'requires_confirmation': True,
            'clinical_validation': False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('dataset', 'manifest', 'probe', 'mirror_probe', 'landmarks', 'output'):
        parser.add_argument('--' + name.replace('_', '-'), type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('Choose a new adaptation directory')
    torch.set_num_threads(4)
    checkpoint = torch.load(args.landmarks, map_location='cpu', weights_only=True)
    if checkpoint['normalization'] != 'physical_proximal_mask150mm_10mm_pad':
        raise ValueError('Unexpected landmark model normalization')
    model = Heatmaps(checkpoint.get('input_channels', 1)).cuda().eval()
    model.load_state_dict(checkpoint['state_dict'], strict=True)
    probes = [json.loads((p / 'probe.json').read_text()) for p in (args.probe, args.mirror_probe)]
    if any(p['labels_sha256'] != sha256(args.manifest) for p in probes) or not probes[1]['input_horizontally_flipped']:
        raise ValueError('Probe source identity differs')
    refs = [{r['pixel_sha256']: r for r in p['results'] if r['polarity'] == 'bone-bright'} for p in probes]
    rows = pd.read_csv(args.manifest)
    cases = []
    for row in rows.itertuples():
        path = args.dataset / row.first_source_path
        image = read_dxa(path)
        femur, pelvis = selected_masks(args.probe, args.mirror_probe, refs[0][row.pixel_sha256], refs[1][row.pixel_sha256], row, image, path)
        masks = np.zeros((23, *image.pixels.shape), bool)
        masks[6], masks[8] = femur, pelvis
        _, points = hip_measurements(masks, image.pixels, image.pixel_mm_x, image.pixel_mm, model)
        if points is None:
            raise ValueError('Missing synthetic-model landmarks in adaptation cohort')
        center, frame, length = anatomical_frame(points[2], points[3], image.pixel_mm_x, image.pixel_mm)
        fixed = neck_roi(femur, points[2], points[3], image.pixel_mm_x, image.pixel_mm)
        reference_path = path.parent.parent / 'segmentations/hip_image.seg.nrrd'
        reference = read_seg_nrrd(reference_path, image.pixels.shape)
        footprint = reference['Femoral_Neck_bone_area'] | reference['Femoral_Neck_soft_tissue_area']
        cases.append({'group': image.study_uid, 'pixel': image.pixel_sha256, 'image': image,
                      'center': center, 'frame': frame, 'length': length, 'reference': footprint, 'fixed': fixed,
                      'target': encode_reference(footprint, center, frame, length, image.pixel_mm_x, image.pixel_mm),
                      'source_sha256': sha256(path), 'reference_sha256': sha256(reference_path)})
    predictions = []
    for case in cases:
        training = [c for c in cases if c['group'] != case['group'] and c['pixel'] != case['pixel']]
        if len(training) < 4:
            raise ValueError('Insufficient study-exclusive ROI donors')
        parameters = np.median([c['target'] for c in training], axis=0)
        image = case['image']
        proposal = proposal_from_template(parameters, case['center'], case['frame'], case['length'],
                                           image.pixel_mm_x, image.pixel_mm, image.pixels.shape)
        overlaps = {}
        for name, candidate in [('fixed', case['fixed']), ('adapted', proposal)]:
            if candidate['status'] != 'proposed':
                overlaps[name] = None
                continue
            raster = np.zeros(image.pixels.shape, np.uint8)
            cv2.fillConvexPoly(raster, np.rint(candidate['polygon']).astype(np.int32), 1)
            overlaps[name] = overlap(case['reference'], raster)
        predictions.append({'case_sha256': hashlib.sha256(case['group'].encode()).hexdigest(),
                            'training_images': len(training), 'held_out_study_used_for_fit': False,
                            'source_sha256': case['source_sha256'], 'reference_sha256': case['reference_sha256'],
                            'proposal': proposal, 'overlap': overlaps})
    report = {'protocol': __doc__, 'images': len(cases), 'dicom_studies': len({c['group'] for c in cases}),
              'mean_dice_fixed': float(np.mean([p['overlap']['fixed']['dice'] for p in predictions if p['overlap']['fixed']])),
              'mean_dice_adapted': float(np.mean([p['overlap']['adapted']['dice'] for p in predictions if p['overlap']['adapted']])),
              'adapted_proposals': sum(p['overlap']['adapted'] is not None for p in predictions),
              'cases': predictions, 'landmark_checkpoint_sha256': sha256(args.landmarks), 'code_sha256': sha256(Path(__file__)),
              'clinical_validation': False, 'automatic_promotions': 0,
              'limitations': ['Previously studied ten published Hologic folders; patient independence not established.',
                              'Scale is device_default; no millimeter accuracy claim.',
                              'Only vendor neck measurement ROI; full anatomical correctness not established.',
                              'All-data template is for future inputs; evaluation above uses study-exclusive templates.']}
    args.output.mkdir(parents=True)
    template = {'schema_version': 1, 'parameters': np.median([c['target'] for c in cases], axis=0).tolist(),
                'landmark_checkpoint_sha256': sha256(args.landmarks), 'clinical_validation': False,
                'reference': 'published Hologic neck measurement ROI', 'training_images': len(cases),
                'code_sha256': sha256(Path(__file__))}
    (args.output / 'template.json').write_text(json.dumps(template, indent=2) + '\n')
    (args.output / 'evaluation.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(json.dumps({k: v for k, v in report.items() if k not in ('cases', 'protocol', 'limitations')}), flush=True)


if __name__ == '__main__':
    main()
