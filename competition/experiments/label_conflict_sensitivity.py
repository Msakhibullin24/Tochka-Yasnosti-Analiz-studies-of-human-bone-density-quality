"""Measure how known label taxonomy conflicts affect saved OOF metrics.

This is a diagnostic exclusion, not relabelling, retraining or a new test set.
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np

from build_release_review import label_taxonomy_conflict
from dxaqc.model import VIOLATION_LABEL
from evaluate_organizer_dataset import CRITERIA, binary_metrics, sha256


def summarize(rows: list[dict], labels: dict[str, dict]) -> dict:
    truth = np.array([int(labels[row['source_path']]['quality_class']) for row in rows])
    predicted = np.array([int(row['quality_pred']) for row in rows])
    score = np.array([float(row['quality_score']) for row in rows])
    if (not np.isin(predicted, (0, 1)).all() or not np.isfinite(score).all()
            or ((score < 0) | (score > 1)).any()):
        raise ValueError('OOF decision or score is outside its contract')
    overall = binary_metrics(truth, predicted, score)
    criteria = {}
    for name in CRITERIA:
        applicable = [(row, labels[row['source_path']]) for row in rows
                      if labels[row['source_path']].get(name) in ('0', '1')]
        y = np.array([int(label[name]) for _, label in applicable])
        decision = np.array([int(VIOLATION_LABEL[name] in row['violation_type'].split(';'))
                             for row, _ in applicable])
        criteria[name] = binary_metrics(y, decision)
    return {'images': len(rows), 'studies': len({row['study'] for row in rows}),
            'overall': overall, 'by_criterion': criteria,
            'criterion_macro_f1': float(np.mean([entry['f1'] for entry in criteria.values()]))}


def audit(oof_path: Path, labels_path: Path) -> dict:
    with oof_path.open(encoding='utf-8-sig', newline='') as stream:
        rows = list(csv.DictReader(stream))
    with labels_path.open(encoding='utf-8-sig', newline='') as stream:
        labels = {row['first_source_path']: row for row in csv.DictReader(stream)
                  if row.get('quality_class') in ('0', '1')}
    if not rows or len(rows) != len(labels) or len({row['source_path'] for row in rows}) != len(rows):
        raise ValueError('OOF must cover every labelled image exactly once')
    folds: dict[str, str] = {}
    for row in rows:
        label = labels.get(row['source_path'])
        if (label is None or row['repeat'] != '0' or row['study'] != label['study_key']
                or row['true_region'] != label['region']
                or row['quality_true'] != label['quality_class']):
            raise ValueError('OOF identity or reference label differs')
        if folds.setdefault(row['study'], row['fold']) != row['fold']:
            raise ValueError('Study occurs in multiple held-out folds')
    conflict = {path: label_taxonomy_conflict(label) for path, label in labels.items()}
    retained = [row for row in rows if not conflict[row['source_path']]]
    if not retained:
        raise ValueError('No nonconflicting labels remain')
    baseline = summarize(rows, labels)
    excluded = summarize(retained, labels)
    return {
        'protocol': 'descriptive sensitivity to excluding complete-label taxonomy conflicts from saved OOF',
        'oof_sha256': sha256(oof_path), 'labels_sha256': sha256(labels_path),
        'excluded_images': len(rows) - len(retained),
        'excluded_reasons': dict(sorted(Counter(reason for reason in conflict.values() if reason).items())),
        'original_labels_edited': False, 'model_changed': False,
        'all_labels': baseline, 'excluding_conflicts': excluded,
        'f1_difference_excluding_minus_all': excluded['overall']['f1'] - baseline['overall']['f1'],
        'macro_f1_difference_excluding_minus_all': (excluded['criterion_macro_f1'] -
                                                     baseline['criterion_macro_f1']),
        'limitations': [
            'Exclusion is a sensitivity analysis only; the three images are not proven mislabeled.',
            'OOF models were trained using the original labels, including the excluded cases.',
            'The reduced cohort is not a new test set, and changes in scores are not model gains.',
            'Patient-disjoint validation is not established by the available anonymized identifiers.',
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--oof', type=Path, required=True)
    parser.add_argument('--labels', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.oof, args.labels)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'excluded_images': result['excluded_images'],
                      'f1_difference_excluding_minus_all': result['f1_difference_excluding_minus_all'],
                      'macro_f1_difference_excluding_minus_all': result['macro_f1_difference_excluding_minus_all']}))


if __name__ == '__main__':
    main()
