"""Research cue for required spine structures, from fixed named-mask detectors.

Only evaluated detections count: a missing feature is not evidence of absent
anatomy. Full ordered L1-L4 candidates are required before using the cue.
The resulting violation is model-estimated and retains an explicit review flag.
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from dxaqc.model import official_violation_type
from evaluate_organizer_dataset import evaluate, sha256
from experiments.verify_metric_candidate import compare


REQUIRED = ('Th12', 'iliac_left', 'iliac_right')


def missing_structures(features, group):
    keys = [f'anat_{name}_present' for name in REQUIRED]
    if group != 'spine' or features.get('anat_lumbar_ordered') != 1 or not all(k in features for k in keys):
        return None
    return [name for name, key in zip(REQUIRED, keys) if features[key] == 0]


def apply_visibility(decision, features, group):
    decision = copy.deepcopy(decision)
    missing = missing_structures(features, group)
    if not missing:
        return decision
    previous = decision['criterion_states']['spine_coverage']
    decision['criterion_states']['spine_coverage'] = {'status': 'fail',
        'basis': 'named_structure_absence_candidate', 'missing_candidates': missing,
        'classifier_state': previous, 'clinical_validation': False}
    if 'spine_coverage' not in decision['violations']:
        decision['violations'].insert(0, 'spine_coverage')
    decision['quality'] = 1
    decision['violation_type_status'] = 'identified'
    decision['decision_reason'] = 'named_structure_absence_candidate_review'
    decision['review_reasons'] = list(dict.fromkeys([*decision['review_reasons'], 'anatomical_visibility_candidate']))
    return decision


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('candidate', 'features', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--labels', type=Path, default=ROOT / 'labels/image_labels.csv')
    parser.add_argument('--baseline', type=Path, default=ROOT / 'reports/quality_review_axis_guard_posthoc_2026_09_29.csv')
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('Choose a new trial directory')
    table = pd.read_csv(args.candidate).sort_values('index').reset_index(drop=True)
    identities = pd.read_csv(args.labels, usecols=['first_source_path', 'pixel_sha256']).set_index('first_source_path')
    source = json.loads(args.features.read_text())
    if source['labels_sha256'] != sha256(args.labels):
        raise ValueError('Anatomical source label identity differs')
    changed, cues = 0, 0
    for i, row in enumerate(table.itertuples()):
        if row.predicted_region != 'spine':
            continue
        values = source['entries'][identities.loc[row.source_path, 'pixel_sha256']]['features']
        states = json.loads(row.criterion_states)
        decision = {'criterion_states': states, 'violations': [c for c, s in states.items() if s['status'] == 'fail'],
                    'quality': int(row.quality_pred), 'violation_type_status': 'identified',
                    'decision_reason': row.decision_reason, 'review_reasons': []}
        decision = apply_visibility(decision, values, 'spine')
        cues += int(bool(missing_structures(values, 'spine')))
        changed += int(decision['quality'] != row.quality_pred)
        for key, value in {'quality_pred': decision['quality'], 'violation_type': official_violation_type(decision['violations']),
                'criterion_states': json.dumps(decision['criterion_states'], sort_keys=True),
                'decision_reason': decision['decision_reason'], 'decision_version': 'research-anatomical-visibility-1'}.items():
            table.at[i, key] = value
    args.output.mkdir(parents=True)
    path = args.output / 'candidate_oof.csv'
    table.to_csv(path, index=False)
    metrics = evaluate(args.labels, path, repeats=2000)
    screen = compare(args.baseline, path, args.labels)
    report = {'protocol': __doc__, 'metrics': metrics, 'screen': screen, 'visibility_cues': cues,
              'binary_decisions_changed': changed, 'input_candidate_sha256': sha256(args.candidate),
              'features_sha256': sha256(args.features), 'code_sha256': sha256(Path(__file__)),
              'clinical_validation': False, 'automatic_promotions': 0,
              'limitations': ['Fixed detector absence is model evidence, not confirmed anatomical absence.',
                              'Already inspected study cohort; not an independent confirmation.']}
    (args.output / 'evaluation.json').write_text(json.dumps(report, indent=2, ensure_ascii=False) + '\n')
    print(json.dumps({'quality': metrics['overall_quality'], 'macro_f1': metrics['criterion_macro_f1'],
                      'visibility_cues': cues, 'binary_decisions_changed': changed}), flush=True)


if __name__ == '__main__':
    main()
