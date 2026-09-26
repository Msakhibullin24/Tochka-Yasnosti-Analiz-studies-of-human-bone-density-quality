"""Compare frozen axis decisions without changing the released inference policy.

The counterfactual uses each outer fold's saved inner-fold threshold, never a
threshold tuned on its test labels. Better label agreement cannot validate the
physical 5-degree requirement or certify anatomical landmarks.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

from dxaqc.model import official_violation_type
from evaluate_organizer_dataset import evaluate, sha256


def counterfactual(row: dict, policy: str) -> dict:
    if policy not in ('learned_axis', 'measured_or_learned_axis'):
        raise ValueError('Unknown counterfactual policy')
    result = dict(row)
    if row['decision_version'] != '5':
        raise ValueError('Expected frozen decision v5 evidence')
    if row['predicted_region'] != 'spine':
        return result
    scores = json.loads(row['criterion_scores'])
    thresholds = json.loads(row['criterion_thresholds'])
    states = json.loads(row['criterion_states'])
    score, threshold = scores['spine_axis'], thresholds['spine_axis']
    if not all(isinstance(v, (int, float)) and math.isfinite(v) for v in (score, threshold)):
        raise ValueError('Invalid saved axis evidence')
    if not 0 <= score <= 1:
        raise ValueError('Invalid saved axis score')
    learned = score >= threshold
    measured = states['spine_axis']['status'] == 'fail'
    failed = learned or measured if policy == 'measured_or_learned_axis' else learned
    states['spine_axis'] = {'status': 'fail' if failed else 'pass',
                            'basis': 'counterfactual_' + policy,
                            'clinical_validation': False}
    types = [key for key, value in states.items() if value['status'] == 'fail']
    binary = float(row['quality_score']) >= float(row['quality_threshold'])
    result.update(quality_pred=str(int(bool(types) or binary)),
                  violation_type=official_violation_type(types),
                  criterion_states=json.dumps(states, sort_keys=True),
                  decision_reason='counterfactual_' + policy)
    return result


def audit(labels: Path, oof: Path, output: Path, bootstrap: int = 1000) -> dict:
    if output.exists():
        raise ValueError('Choose a new audit directory')
    baseline = evaluate(labels, oof, repeats=bootstrap)  # validate identities and folds first
    with oof.open(newline='') as stream:
        rows = list(csv.DictReader(stream))
    candidates = {policy: [counterfactual(row, policy) for row in rows]
                  for policy in ('learned_axis', 'measured_or_learned_axis')}
    output.mkdir(parents=True)
    metrics = {'released_v5': baseline}
    for policy, candidate in candidates.items():
        path = output / f'{policy}_oof.csv'
        with path.open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(candidate)
        metrics[policy] = evaluate(labels, path, repeats=bootstrap)
        metrics[policy]['protocol'] = 'counterfactual on frozen study-held-out scores; not released inference'
    report = {'scope': 'diagnosis only; no weights, source labels or released decisions changed',
              'eligible_for_clinical_release': False,
              'labels_sha256': sha256(labels), 'oof_sha256': sha256(oof),
              'policies': metrics,
              'limitations': [
                  'Policies inspected on these folds need a fresh independent test before selection.',
                  'A learned label does not establish a measured angle above 5 degrees.',
                  'None of these policies supplies Th12 numbering, projection or source ROI validation.']}
    (output / 'audit.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('labels', 'oof', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--bootstrap', type=int, default=1000)
    args = parser.parse_args()
    report = audit(args.labels, args.oof, args.output, args.bootstrap)
    print(json.dumps({name: {'quality_f1': value['overall_quality']['f1'],
                            'axis': value['by_criterion']['spine_axis']['tn_fp_fn_tp'],
                            'untyped': value['predicted_positive_without_type']}
                      for name, value in report['policies'].items()}))
