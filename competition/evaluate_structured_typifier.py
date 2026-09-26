"""Evaluate a learned normal/type decision on frozen baseline study folds.

Released predictions and labels are not modified. Every secondary model sees
only other studies. Architecture choices compared here require fresh data to
establish unbiased post-selection performance.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from dxaqc.model import CRITERIA, group_of, official_violation_type
from evaluate_organizer_dataset import binary_metrics, evaluate, sha256
from source_integrity import inspect_sources
from structured_typifier import StructuredTypifier, target
from train import build_table, HERE


def paired_f1_interval(rows, candidates, repeats=1000):
    if len(rows) != len(candidates) or any(a['source_path'] != b['source_path'] for a, b in zip(rows, candidates)):
        raise ValueError('Paired predictions must have matching image identities')
    truth = np.array([int(row['quality_true']) for row in rows])
    original = np.array([int(row['quality_pred']) for row in rows])
    predicted = np.array([int(row['quality_pred']) for row in candidates])
    studies = np.array([row['study'] for row in rows])
    units = [np.flatnonzero(studies == study) for study in np.unique(studies)]
    rng = np.random.default_rng(17)
    delta = []
    for _ in range(repeats):
        sample = np.concatenate([units[i] for i in rng.integers(len(units), size=len(units))])
        delta.append(binary_metrics(truth[sample], predicted[sample])['f1']-
                     binary_metrics(truth[sample], original[sample])['f1'])
    return list(map(float, np.percentile(delta, [2.5, 97.5])))


def apply_candidate(row, prediction, policy):
    if policy not in ('untyped_only', 'all_images'):
        raise ValueError('Unknown candidate policy')
    result = dict(row)
    if policy == 'untyped_only' and not (row['quality_pred'] == '1' and not row['violation_type']):
        return result
    if not set(prediction['codes']) <= CRITERIA[group_of(row['predicted_region'])].keys():
        raise ValueError('Prediction contains a nonofficial or wrong-region type')
    result.update(quality_pred=str(int(bool(prediction['codes']))),
                  violation_type=official_violation_type(prediction['codes']),
                  decision_version='experimental-structured-1',
                  decision_reason='experimental_structured_'+('typed' if prediction['codes'] else 'normal'),
                  typifier_evidence=json.dumps(prediction, ensure_ascii=False, sort_keys=True))
    return result


def run(dataset, labels_path, oof_path, output, bootstrap=1000):
    if output.exists():
        raise ValueError('Choose a new experiment directory')
    baseline = evaluate(labels_path, oof_path, repeats=bootstrap)
    audit = inspect_sources([('organiser', labels_path, dataset)])
    import hashlib
    fingerprint = hashlib.sha256(json.dumps(
        [(entry['path'], entry['pixel_sha256_current']) for entry in audit['entries']]).encode()).hexdigest()
    labels = pd.read_csv(labels_path)
    labels = labels[labels.quality_class.notna()].reset_index(drop=True)
    features, embedding, _ = build_table(dataset, labels, HERE/'.cache', fingerprint)
    with labels_path.open(newline='') as stream:
        reference = {r['first_source_path']: r for r in csv.DictReader(stream)
                     if r['quality_class'] in ('0', '1')}
    with oof_path.open(newline='') as stream:
        rows = list(csv.DictReader(stream))
    index_by_path = {str(row.first_source_path): index for index, row in enumerate(labels.itertuples())}
    targets = [target(reference[path]) for path in labels.first_source_path]
    output.mkdir(parents=True)
    reports = {'released_v5': baseline}
    for family in ('logistic', 'forest'):
        predictions = {}
        fold_exclusions = []
        for fold in sorted({row['fold'] for row in rows}):
            held_out_studies = {row['study'] for row in rows if row['fold'] == fold}
            for group in ('spine', 'hip'):
                training = [i for i, row in enumerate(labels.itertuples())
                            if row.study_key not in held_out_studies and group_of(row.region) == group
                            and targets[i] is not None]
                test_rows = [row for row in rows if row['fold'] == fold
                             and group_of(row['predicted_region']) == group]
                testing = [index_by_path[row['source_path']] for row in test_rows]
                assert not held_out_studies & set(labels.iloc[training].study_key)
                fitted = StructuredTypifier(group, family).fit(
                    [features[i] for i in training], embedding[training], [targets[i] for i in training])
                for row, prediction in zip(test_rows, fitted.predict([features[i] for i in testing], embedding[testing])):
                    predictions[row['source_path']] = prediction
                fold_exclusions.append({'fold': fold, 'group': group, 'training_images': len(training),
                                        'test_images': len(testing)})
        for policy in ('untyped_only', 'all_images'):
            candidates = [apply_candidate(row, predictions[row['source_path']], policy) for row in rows]
            path = output/f'{family}_{policy}_oof.csv'
            fields = list(rows[0]) + ['typifier_evidence']
            with path.open('w', newline='') as stream:
                writer = csv.DictWriter(stream, fieldnames=fields); writer.writeheader(); writer.writerows(candidates)
            reports[family+'_'+policy] = evaluate(labels_path, path, repeats=bootstrap)
            reports[family+'_'+policy]['protocol'] = 'experimental secondary classifier on baseline outer study folds'
            reports[family+'_'+policy]['paired_delta_f1_ci95'] = paired_f1_interval(rows, candidates, bootstrap)
        for group in ('spine', 'hip'):
            selected = [i for i, row in enumerate(labels.itertuples())
                        if group_of(row.region) == group and targets[i] is not None]
            fitted = StructuredTypifier(group, family).fit(
                [features[i] for i in selected], embedding[selected], [targets[i] for i in selected])
            joblib.dump(fitted, output/f'{family}_{group}.joblib')
        print(f'Finished structured {family}', flush=True)
    report = {'scope': 'research alternatives; shipped models and source labels unchanged',
              'labels_sha256': sha256(labels_path), 'baseline_oof_sha256': sha256(oof_path),
              'source_identity': {k: v for k, v in audit.items() if k != 'entries'},
              'excluded_reference_targets': sum(t is None for t in targets), 'policies': reports,
              'fold_training_scope': fold_exclusions,
              'evaluation_code_sha256': sha256(Path(__file__)),
              'typifier_code_sha256': sha256(HERE/'structured_typifier.py'),
              'model_sha256': {path.name: sha256(path) for path in sorted(output.glob('*.joblib'))},
              'clinical_validation': False,
              'limitations': ['A predicted axis type does not establish a physical angle above 5 degrees.',
                              'False positives and false negatives remain possible despite complete categorical answers.',
                              'Comparison and architecture selection on these folds are not an independent validation.',
                              'Targets with conflicting or incomplete reference criteria are excluded from training only.']}
    (output/'evaluation.json').write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n')
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('dataset', 'labels', 'oof', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--bootstrap', type=int, default=1000)
    args = parser.parse_args()
    result = run(args.dataset, args.labels, args.oof, args.output, args.bootstrap)
    print(json.dumps({name: {'f1': value['overall_quality']['f1'],
                            'macro_type_f1': value['criterion_macro_f1'],
                            'untyped': value['predicted_positive_without_type']}
                      for name, value in result['policies'].items()}))
