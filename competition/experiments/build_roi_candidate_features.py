"""Build QC inputs from checked predicted anatomy and own neck-ROI geometry.

No QC target values are read. Spine inputs remain the measured 145-column
numbered-axis experiment; hip inputs add mask/landmark/ROI relationships.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from dxaqc.dicom_io import read_dxa
from gpu_research.common import digest, fingerprint
from source_integrity import inspect_sources
from experiments.hip_mask_features import hip_measurements
from experiments.transfer_camera_projection import selected_masks
from experiments.train_silhouette_heatmaps import Heatmaps
from experiments.neck_roi_from_mask import neck_roi
from experiments.adapt_native_neck_roi import anatomical_frame, proposal_from_template
from experiments.roi_candidate_features import measure_roi


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('spine_features', 'probe', 'mirror_probe', 'landmarks', 'template', 'dataset', 'output'):
        parser.add_argument('--' + name.replace('_', '-'), type=Path, required=True)
    parser.add_argument('--labels', type=Path, default=ROOT / 'labels/image_labels.csv')
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('Choose a new feature directory')
    report = json.loads(args.spine_features.read_text())
    if (report['labels_sha256'] != digest(args.labels) or report['source_fingerprint'] !=
            fingerprint(inspect_sources([('organiser', args.labels, args.dataset)])['entries'])):
        raise ValueError('Anatomical source identities differ')
    probes = [json.loads((p / 'probe.json').read_text()) for p in (args.probe, args.mirror_probe)]
    if (any(p['labels_sha256'] != digest(args.labels) for p in probes) or not probes[1]['input_horizontally_flipped']
            or probes[0]['checkpoint_sha256'] != probes[1]['checkpoint_sha256']):
        raise ValueError('Anatomy probes differ')
    refs = [{r['pixel_sha256']: r for r in p['results'] if r['polarity'] == 'bone-bright'} for p in probes]
    checkpoint = torch.load(args.landmarks, map_location='cpu', weights_only=True)
    if checkpoint['normalization'] != 'physical_proximal_mask150mm_10mm_pad':
        raise ValueError('Unexpected landmark model normalization')
    template = json.loads(args.template.read_text())
    if template['landmark_checkpoint_sha256'] != digest(args.landmarks):
        raise ValueError('Native ROI template landmark checkpoint differs')
    torch.set_num_threads(4)
    model = Heatmaps(checkpoint.get('input_channels', 1)).cuda().eval()
    model.load_state_dict(checkpoint['state_dict'], strict=True)
    rows = pd.read_csv(args.labels, usecols=['first_source_path', 'pixel_sha256', 'region'])
    measurements = []
    for row in rows.itertuples():
        if not row.region.startswith('hip') or row.pixel_sha256 not in refs[0]:
            continue
        path = args.dataset / row.first_source_path
        image = read_dxa(path)
        femur, pelvis = selected_masks(args.probe, args.mirror_probe, refs[0][row.pixel_sha256], refs[1][row.pixel_sha256], row, image, path)
        masks = np.zeros((23, *image.pixels.shape), bool)
        masks[6], masks[8] = femur, pelvis
        features, points = hip_measurements(masks, image.pixels, image.pixel_mm_x, image.pixel_mm, model)
        proposal = {'status': 'unavailable', 'reason': 'landmarks_unavailable'}
        if points is not None:
            proposal = neck_roi(femur, points[2], points[3], image.pixel_mm_x, image.pixel_mm)
            if proposal['status'] == 'proposed':
                center, frame, length = anatomical_frame(points[2], points[3], image.pixel_mm_x, image.pixel_mm)
                adapted = proposal_from_template(template['parameters'], center, frame, length, image.pixel_mm_x,
                                                  image.pixel_mm, image.pixels.shape)
                if adapted['status'] == 'proposed':
                    proposal = adapted
        extra, assessment = measure_roi(proposal, femur, pelvis, points, image.pixel_mm_x, image.pixel_mm)
        features.update(extra)
        report['entries'][row.pixel_sha256] = {'features': features}
        measurements.append({'historical_pixel_sha256': row.pixel_sha256, 'assessment': assessment})
    report['columns_by_group']['hip'] = sorted({key for row in measurements for key in report['entries'][row['historical_pixel_sha256']]['features']})
    report.update(hip_checkpoint_sha256=probes[0]['checkpoint_sha256'],
                  hip_landmark_checkpoint_sha256=digest(args.landmarks), native_roi_template_sha256=digest(args.template),
                  roi_geometry_code_sha256=digest(Path(__file__).with_name('roi_candidate_features.py')))
    args.output.mkdir(parents=True)
    (args.output / 'features.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    (args.output / 'measurements.json').write_text(json.dumps({'protocol': __doc__, 'images': len(measurements),
        'code_sha256': digest(Path(__file__)), 'cases': measurements, 'clinical_validation': False,
        'hard_failure_to_qc_defect_rule': False}, indent=2) + '\n')
    print(json.dumps({'hip_images': len(measurements), 'hip_columns': len(report['columns_by_group']['hip']),
                      'spine_columns': len(report['columns_by_group']['spine'])}), flush=True)


if __name__ == '__main__':
    main()
