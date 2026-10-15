"""Three fixed physical axis hypotheses from model-derived L1--L4 body cores.

Global OLS, endpoint direction and area-weighted global OLS are evaluated without
changing the TZ threshold of 5 degrees. These are estimates, not ground truth;
no expert QC values or comments are read to construct them. Local pair angles
are recorded as descriptive curvature, never substituted for global inclination.
"""
import argparse
import copy
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy.ndimage import distance_transform_edt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from dxaqc.dicom_io import read_dxa
from gpu_research.common import digest
from experiments.audit_automatic_geometry import references, load_masks
from experiments.automatic_anatomy_geometry import body_component
from experiments.mask_anatomy_features import largest_component

EXECUTION_SOURCE = Path(__file__).read_text()
EXECUTION_SHA256 = digest(Path(__file__))


def body_axes(masks, sx, sy):
    raw = [largest_component(masks[c]) for c in (6, 5, 4, 3)]
    if any(m.sum() < 4 for m in raw):
        return {'status': 'unavailable', 'reason': 'missing_numbered_lumbar_mask'}
    radius = .35*float(np.median([distance_transform_edt(m, sampling=(sy, sx)).max() for m in raw]))
    bodies = [body_component(m, sx, sy, radius) for m in raw]
    if any(m.sum() < 4 for m in bodies):
        return {'status': 'unavailable', 'reason': 'body_core_not_resolved'}
    centers = np.array([[np.where(m)[1].mean()*sx, np.where(m)[0].mean()*sy] for m in bodies])
    differences = np.diff(centers, axis=0)
    if np.any(differences[:, 1] <= sy):
        return {'status': 'unavailable', 'reason': 'unordered_or_degenerate_body_centers'}
    area = np.array([m.sum()*sx*sy for m in bodies])
    angles = {
        'global': float(np.degrees(np.arctan(np.polyfit(centers[:, 1], centers[:, 0], 1)[0]))),
        'endpoint': float(np.degrees(np.arctan2(*(centers[-1]-centers[0])))),
        'weighted_global': float(np.degrees(np.arctan(np.polyfit(centers[:, 1], centers[:, 0], 1, w=np.sqrt(area))[0])))}
    slope, intercept = np.polyfit(centers[:, 1], centers[:, 0], 1)
    return {'status': 'evaluated', 'angles_degrees': angles, 'centers_mm': centers.tolist(),
            'area_mm2': area.tolist(), 'opening_radius_mm': radius,
            'pair_angles_degrees': np.degrees(np.arctan2(differences[:, 0], differences[:, 1])).tolist(),
            'global_residual_mm': float(np.sqrt(np.mean((centers[:, 0] - slope*centers[:, 1] - intercept)**2))),
            'clinical_validation': False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('base', 'probe', 'dataset', 'output'): parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--labels', type=Path, default=ROOT/'labels/image_labels.csv')
    args = parser.parse_args()
    if args.output.exists(): raise ValueError('Choose a new feature output directory')
    base, (meta, refs) = json.loads(args.base.read_text()), references(args.probe)
    if base['labels_sha256'] != digest(args.labels) or meta['labels_sha256'] != digest(args.labels):
        raise ValueError('Anatomy and source label identities differ')
    labels = pd.read_csv(args.labels, usecols=['first_source_path', 'pixel_sha256', 'region'])
    variants = {name: copy.deepcopy(base) for name in ('global', 'endpoint', 'weighted_global')}
    audit = []
    for row in labels.itertuples():
        if row.region != 'spine' or row.pixel_sha256 not in refs: continue
        path = args.dataset/row.first_source_path
        image = read_dxa(path)
        masks = load_masks(args.probe, refs[row.pixel_sha256], path, image, row.pixel_sha256)
        measured = body_axes(masks, image.pixel_mm_x, image.pixel_mm)
        audit.append({'historical_pixel_sha256': row.pixel_sha256, 'decoded_pixel_sha256': image.pixel_sha256, **measured})
        for name, report in variants.items():
            entry = report['entries'][row.pixel_sha256]
            angle = measured.get('angles_degrees', {}).get(name)
            entry['numbered_axis_degrees'] = angle
            if angle is not None:
                entry['features']['anat_lumbar_angle_abs_deg'] = abs(angle)
                entry['features']['anat_lumbar_line_residual_mm'] = measured['global_residual_mm']
                for i in range(4):
                    entry['features'][f'anat_L{i+1}_center_x'] = measured['centers_mm'][i][0]/(image.pixel_mm_x*image.pixels.shape[1])
                    entry['features'][f'anat_L{i+1}_center_y'] = measured['centers_mm'][i][1]/(image.pixel_mm*image.pixels.shape[0])
                for i, pair in enumerate(measured['pair_angles_degrees'], 1):
                    entry['features'][f'anat_lumbar_pair_{i}_angle_abs_deg'] = abs(pair)
                    entry['features'][f'anat_lumbar_pair_{i}_height_mm'] = measured['centers_mm'][i][1] - measured['centers_mm'][i-1][1]
    args.output.mkdir(parents=True)
    sources = {'base_sha256': digest(args.base), 'probe_sha256': digest(args.probe/'probe.json'),
               'code_sha256': EXECUTION_SHA256, 'geometry_code_sha256': digest(Path(__file__).with_name('automatic_anatomy_geometry.py'))}
    for name, report in variants.items():
        report['body_axis_hypothesis'] = {'method': name, 'sources': sources, 'qc_targets_used': False,
                                          'physical_threshold_degrees': 5., 'clinical_validation': False}
        (args.output/name).mkdir()
        (args.output/name/'features.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    (args.output/'measurements.json').write_text(json.dumps({'protocol': __doc__, 'sources': sources, 'cases': audit}, indent=2)+'\n')
    (args.output/'executed_spine_body_axis_candidates.py').write_text(EXECUTION_SOURCE)
    print(json.dumps({'spine_images': len(audit), 'available': sum(r['status']=='evaluated' for r in audit)}))


if __name__ == '__main__': main()
