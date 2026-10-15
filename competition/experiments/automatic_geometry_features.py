"""Append fixed model-derived body/ROI measurements, without reading QC targets.

Missing measurements have explicit availability flags. This is a research
feature hypothesis; measurements must not be converted into hard QC rules.
"""
import argparse
import copy
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from gpu_research.common import digest


def measurements(region, geometry):
    if region == 'spine':
        body = geometry.get('th12', {})
        fraction = body.get('visible_fraction_estimate')
        return {'anat_body_fraction_available': int(fraction is not None),
                'anat_body_visible_fraction': fraction or 0.,
                'anat_body_touches_top': int(body.get('touches_superior_frame', False)),
                'anat_body_height_mm': body.get('observed_body_extent_mm') or 0.,
                'anat_body_reference_height_mm': body.get('reference_height_mm') or 0.}
    gt = geometry.get('greater_trochanter_compartment', {})
    ends = geometry.get('soft_tissue_sides', {}).get('ends', [])
    ratios = [end['predicted_muscle_nonbone_pixels']/max(1, end['roi_pixels']) for end in ends]
    return {'anat_relationships_available': int(geometry.get('status') == 'evaluated'),
            'anat_gt_compartment_area_mm2': gt.get('area_mm2', 0.),
            'anat_gt_compartment_roi_fraction': gt.get('roi_overlap_fraction_of_compartment', 0.),
            'anat_muscle_end_min_fraction': min(ratios, default=0.),
            'anat_muscle_end_max_fraction': max(ratios, default=0.),
            'anat_pelvic_roi_fraction': geometry.get('pelvic_overlap', {}).get('fraction_of_roi', 0.)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', type=Path, required=True)
    parser.add_argument('--geometry', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists(): raise ValueError('Choose a new feature output directory')
    base, audit = json.loads(args.base.read_text()), json.loads(args.geometry.read_text())
    if base['labels_sha256'] != audit['source_manifest_sha256']:
        raise ValueError('Geometry and base feature label identities differ')
    result = copy.deepcopy(base)
    for record in audit['results']:
        key = record['historical_pixel_sha256']
        if key not in result['entries']: raise ValueError('Geometry identity missing from base')
        group = 'spine' if record['region'] == 'spine' else 'hip'
        extra = measurements(record['region'], record['geometry'])
        if set(extra) & set(result['entries'][key]['features']): raise ValueError('Feature name collision')
        result['entries'][key]['features'].update(extra)
        result['columns_by_group'][group] = sorted(set(result['columns_by_group'][group]) | set(extra))
    result['automatic_geometry_sources'] = {'base_sha256': digest(args.base),
        'audit_sha256': digest(args.geometry), 'code_sha256': digest(Path(__file__)),
        'qc_targets_used': False, 'hypothesis': __doc__,
        'geometry_code_sha256': audit['geometry_code_sha256']}
    args.output.mkdir(parents=True)
    (args.output/'features.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    (args.output/'executed_automatic_geometry_features.py').write_text(Path(__file__).read_text())
    print(json.dumps({'images': len(audit['results']), 'columns_by_group': {k:len(v) for k,v in result['columns_by_group'].items()}}))


if __name__ == '__main__': main()
