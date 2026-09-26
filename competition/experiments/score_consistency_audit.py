"""Descriptive OOF score audit; never fits a calibrator or changes release decisions."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import brier_score_loss, roc_auc_score

from evaluate_organizer_dataset import sha256


def reliability_bins(truth: np.ndarray, score: np.ndarray, count: int = 5) -> list[dict]:
    if len(truth) != len(score) or not len(truth) or count < 2:
        raise ValueError('truth and score must be nonempty and aligned; count must be at least 2')
    bins = []
    for index in range(count):
        low, high = index / count, (index + 1) / count
        selected = (score >= low) & (score <= high if index == count - 1 else score < high)
        bins.append({'from': low, 'to': high, 'n': int(selected.sum()),
                     'mean_score': float(score[selected].mean()) if selected.any() else None,
                     'observed_rate': float(truth[selected].mean()) if selected.any() else None})
    return bins


def audit(oof_path: Path, labels_path: Path) -> dict:
    with oof_path.open(encoding='utf-8-sig', newline='') as stream:
        rows = list(csv.DictReader(stream))
    with labels_path.open(encoding='utf-8-sig', newline='') as stream:
        labels = {row['first_source_path']: row for row in csv.DictReader(stream)
                  if row.get('quality_class') in ('0', '1')}
    if not rows or len(rows) != len(labels) or len({row['source_path'] for row in rows}) != len(rows):
        raise ValueError('OOF must contain each labelled image exactly once')
    groups: dict[str, str] = {}
    for row in rows:
        label = labels.get(row['source_path'])
        if (label is None or row['repeat'] != '0' or row['study'] != label['study_key']
                or row['true_region'] != label['region'] or row['quality_true'] != label['quality_class']):
            raise ValueError('OOF identity or label does not match the reference')
        if groups.setdefault(row['study'], row['fold']) != row['fold']:
            raise ValueError('Study occurs in multiple held-out folds')
    truth = np.array([int(row['quality_true']) for row in rows])
    decision = np.array([int(row['quality_pred']) for row in rows])
    score = np.array([float(row['quality_score']) for row in rows])
    if (not np.isfinite(score).all() or ((score < 0) | (score > 1)).any()
            or any(value not in (0, 1) for value in decision)):
        raise ValueError('Scores or decisions are outside the contract')
    bins = reliability_bins(truth, score)
    ece = sum(item['n'] / len(rows) * abs(item['mean_score'] - item['observed_rate'])
              for item in bins if item['n'])
    prevalence = float(truth.mean())
    return {
        'protocol': 'descriptive audit of saved one-repeat study-held-out OOF; no calibration fitted',
        'oof_sha256': sha256(oof_path), 'labels_sha256': sha256(labels_path),
        'images': len(rows), 'studies': len(groups), 'positives': int(truth.sum()),
        'mean_score': float(score.mean()), 'observed_prevalence': prevalence,
        'roc_auc': float(roc_auc_score(truth, score)),
        'brier': float(brier_score_loss(truth, score)),
        'constant_prevalence_brier': float(brier_score_loss(truth, np.full(len(truth), prevalence))),
        'ece_5_equal_width_bins': float(ece), 'reliability_bins': bins,
        'decision_score_crossings_at_0_5': {
            'positive_decision_below_0_5': int(((decision == 1) & (score < .5)).sum()),
            'normal_decision_at_least_0_5': int(((decision == 0) & (score >= .5)).sum()),
            'note': '0.5 is only a descriptive reference; official decisions use typed criteria and learned thresholds.'},
        'limitations': [
            'OOF was previously inspected and is not an independent clinical test.',
            'Five-bin ECE is descriptive on 249 images; no calibration confidence interval is claimed.',
            'A post-hoc calibration fitted on these same predictions would leak evaluation labels.',
            'The shipped model and its final inference threshold are not changed by this audit.',
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--oof', type=Path, required=True)
    parser.add_argument('--labels', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    report = audit(args.oof, args.labels)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'images': report['images'], 'brier': report['brier'],
                      'ece_5_equal_width_bins': report['ece_5_equal_width_bins']}, ensure_ascii=False))


if __name__ == '__main__':
    main()
