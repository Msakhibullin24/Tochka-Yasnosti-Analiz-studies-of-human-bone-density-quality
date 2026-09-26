"""Evaluate conditional typing on saved study-held-out model distributions.

No fitting or new expert labels. This post-hoc experiment is not a new held-out
test; only the detector's binary decisions and physical checks are immutable.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from conditional_typifier import apply_conditional_type
from evaluate_organizer_dataset import evaluate, sha256


def paired_candidates(rows, secondary):
    indexed = {row['source_path']: row for row in secondary}
    identities = {row['source_path'] for row in rows}
    if len(indexed) != len(secondary) or len(identities) != len(rows) or identities != set(indexed):
        raise ValueError('Duplicate or mismatched image identities')
    candidates = []
    for row in rows:
        source = indexed[row['source_path']]
        if any(source.get(key) != row.get(key) for key in ('study', 'fold', 'quality_true', 'predicted_region')):
            raise ValueError('Secondary prediction has mismatched fold, study or reference')
        candidates.append(apply_conditional_type(row, json.loads(source['typifier_evidence'])))
    return candidates


def read_rows(path):
    with path.open(newline='') as stream:
        return list(csv.DictReader(stream))


def run(labels, baseline, secondary, output, bootstrap=1000):
    if output.exists():
        raise ValueError('Choose a new experiment directory')
    reference = evaluate(labels, baseline, repeats=bootstrap)
    rows = read_rows(baseline)
    candidates = paired_candidates(rows, read_rows(secondary))
    output.mkdir(parents=True)
    path = output / 'conditional_oof.csv'
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]) + ['conditional_type_evidence'])
        writer.writeheader()
        writer.writerows(candidates)
    metrics = evaluate(labels, path, repeats=bootstrap)
    for value in metrics['by_criterion'].values():
        value['score_scope'] = 'saved baseline criterion head; not conditional type confidence'
    evidence = [json.loads(row['conditional_type_evidence']) for row in candidates
                if row.get('conditional_type_evidence')]
    report = {'scope': 'post-hoc study-held-out experiment without new labels; not deployed',
              'labels_sha256': sha256(labels), 'baseline_sha256': sha256(baseline),
              'secondary_sha256': sha256(secondary),
              'decoder_sha256': sha256(Path(__file__).with_name('conditional_typifier.py')),
              'evaluation_code_sha256': sha256(Path(__file__)),
              'baseline': reference, 'candidate': metrics,
              'binary_decisions_changed': sum(a['quality_pred'] != b['quality_pred'] for a, b in zip(rows, candidates)),
              'type_candidates': sum(e['status'] == 'candidate' for e in evidence),
              'secondary_normal_disagreements': sum(e['secondary_normal_disagreement'] for e in evidence),
              'clinical_validation': False, 'release_modified': False,
              'limitations': ['Conditional type remains a hypothesis and cannot become a clinical pass.',
                             'Geometry coverage, numbering, projection and source ROI are not supplied by typing.',
                             'Model selection on previously inspected folds is not an independent test.',
                             'No eligible type leaves the alarm untyped instead of inventing a label.']}
    (output / 'evaluation.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('labels', 'baseline', 'secondary', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--bootstrap', type=int, default=1000)
    args = parser.parse_args()
    report = run(args.labels, args.baseline, args.secondary, args.output, args.bootstrap)
    print(json.dumps({'f1': report['candidate']['overall_quality']['f1'],
                      'macro_type_f1': report['candidate']['criterion_macro_f1'],
                      'untyped': report['candidate']['predicted_positive_without_type'],
                      'binary_changed': report['binary_decisions_changed']}))
