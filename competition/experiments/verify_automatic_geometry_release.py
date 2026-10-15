"""Compare final-fit delivery answers and report geometry availability.

This verifies regression and execution, never accuracy on fitting images.
"""
import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path


def rows(path):
    with path.open(encoding='utf-8-sig', newline='') as stream:
        return list(csv.DictReader(stream))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--previous', type=Path, required=True)
    parser.add_argument('--current', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    old, new = rows(args.previous/'results_extended.csv'), rows(args.current/'results_extended.csv')
    fields = ('image_uid', 'quality_class', 'violation_type', 'quality_prob', 'processing_status', 'anatomical_region')
    answers = lambda values: Counter(tuple(row[k] for k in fields) for row in values)
    if len(old) != len(new) or answers(old) != answers(new):
        raise ValueError('Geometry integration changed final QC answers')
    geometry = json.loads((args.current/'automatic_geometry_report.json').read_text())
    roi_counts = Counter()
    for row in new:
        if row['duplicate_of']: continue
        learned = json.loads(row['learned_anatomy'])
        if 'density_based_roi' in learned:
            roi_counts['spine_unique_images'] += 1
            roi_counts['spine_roi_proposals'] += len(learned['roi_proposals'])
            roi_counts['spine_density_roi_proposals'] += sum(
                p['method'] == 'named_mask_and_projection_density_minima' for p in learned['roi_proposals'])
        else:
            roi_counts['hip_unique_images'] += 1
            roi_counts['hip_roi_proposed'] += learned['neck_roi_assessment']['status'] == 'proposed'
    unique, counts = {}, Counter()
    for item in geometry['records']:
        key = item['decoded_pixel_sha256']
        if key in unique: raise ValueError('Unique decoded geometry identity is duplicated')
        unique[key] = item
        data = item['geometry']
        counts['unique_images'] += 1
        if item['region'] == 'spine':
            counts['spine'] += 1
            counts['Th12_'+data['th12'].get('outcome', 'unavailable')] += 1
            for crest in data['iliac_crests']: counts[crest['id']+'_'+crest['outcome']] += 1
        else:
            counts['hip'] += 1
            counts['hip_geometry_'+data['status']] += 1
            if data['status'] == 'evaluated':
                counts['soft_tissue_both_ends_supported'] += data['soft_tissue_sides']['predicted_soft_tissue_at_both_ends']
                counts['estimated_GT_compartment_excluded'] += data['greater_trochanter_compartment']['estimated_compartment_excluded']
        if data.get('clinical_validation') is not False or data.get('affects_qc_classifier') is not False:
            raise ValueError('Geometry scope must remain explicit')
    audit = {'protocol': __doc__, 'files': len(new), 'qc_answers_identical': True,
             'compared_fields': fields, 'unique_geometry_counts': dict(counts),
             **roi_counts,
             'clinical_validation': False, 'accuracy_evaluation': False,
             'requirements': json.loads((args.current/'requirements.json').read_text()),
             'source_sha256': {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in
                              (args.previous/'results_extended.csv', args.current/'results_extended.csv',
                               args.current/'automatic_geometry_report.json', Path(__file__))},
             'limitations': ['Detection/measurement counts do not establish anatomical accuracy.',
                            'Final heads were fitted on all available QC labels.']}
    args.output.write_text(json.dumps(audit, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    print(json.dumps({'files': len(new), 'qc_answers_identical': True, 'unique_geometry_counts': dict(counts)}))


if __name__ == '__main__': main()
