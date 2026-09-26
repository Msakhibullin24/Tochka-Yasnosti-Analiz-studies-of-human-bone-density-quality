"""Score study-held-out predictions against the organizer's image labels.

This evaluates the saved OOF decision path. It does not evaluate shipped weights
on an independent dataset; the three supplied test DICOM have no answer key.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from dxaqc.model import VIOLATION_LABEL

CRITERIA = ('spine_coverage', 'spine_axis', 'spine_artifact',
            'hip_position_rotation', 'hip_roi_coverage')


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def binary_metrics(y: np.ndarray, pred: np.ndarray, score: np.ndarray | None = None) -> dict:
    tn = int(((y == 0) & (pred == 0)).sum())
    fp = int(((y == 0) & (pred == 1)).sum())
    fn = int(((y == 1) & (pred == 0)).sum())
    tp = int(((y == 1) & (pred == 1)).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    specificity = tn / (tn + fp) if tn + fp else 0.0
    result = {
        'n': len(y), 'positives': int(y.sum()), 'negatives': int(len(y) - y.sum()),
        'tn_fp_fn_tp': [tn, fp, fn, tp], 'accuracy': (tn + tp) / len(y),
        'precision': precision, 'sensitivity': recall, 'specificity': specificity,
        'f1': 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0,
        'balanced_accuracy': (recall + specificity) / 2 if tp + fn and tn + fp else None,
    }
    if score is not None:
        result['roc_auc'] = float(roc_auc_score(y, score)) if len(np.unique(y)) == 2 else None
        result['average_precision'] = (float(average_precision_score(y, score))
                                       if len(np.unique(y)) == 2 else None)
    return result


def cluster_ci(y: np.ndarray, pred: np.ndarray, groups: np.ndarray,
               score: np.ndarray | None = None, repeats: int = 1000) -> dict:
    rng = np.random.default_rng(17)
    indices = [np.flatnonzero(groups == group) for group in np.unique(groups)]
    values: dict[str, list[float]] = {'f1': [], 'sensitivity': [], 'specificity': [],
                                      'roc_auc': [], 'average_precision': []}
    for _ in range(repeats):
        sample = np.concatenate([indices[i] for i in rng.integers(len(indices), size=len(indices))])
        metrics = binary_metrics(y[sample], pred[sample], score[sample] if score is not None else None)
        for key in values:
            if metrics.get(key) is not None:
                values[key].append(metrics[key])
    return {key: {'low': float(np.percentile(vals, 2.5)),
                  'high': float(np.percentile(vals, 97.5)),
                  'defined_resamples': len(vals)}
            for key, vals in values.items() if vals}


def evaluate(labels_path: Path, oof_path: Path, repeats: int = 1000) -> dict:
    with labels_path.open(newline='') as stream:
        labels = {row['first_source_path']: row for row in csv.DictReader(stream)
                  if row['quality_class'] in ('0', '1')}
    with oof_path.open(newline='') as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != len(labels) or len({row['source_path'] for row in rows}) != len(rows):
        raise ValueError('OOF rows are incomplete or duplicated relative to labelled images')
    fold_by_study: dict[str, str] = {}
    for row in rows:
        label = labels.get(row['source_path'])
        if label is None or row['study'] != label['study_key'] or row['true_region'] != label['region']:
            raise ValueError('OOF identity, study or region differs from reference label')
        if row['quality_true'] != label['quality_class'] or row['repeat'] != '0':
            raise ValueError('OOF quality label or repeat differs from expected protocol')
        prior_fold = fold_by_study.setdefault(row['study'], row['fold'])
        if prior_fold != row['fold']:
            raise ValueError('Study is split across test folds')
    quality = np.array([int(row['quality_true']) for row in rows])
    predicted = np.array([int(row['quality_pred']) for row in rows])
    scores = np.array([float(row['quality_score']) for row in rows])
    if not np.isfinite(scores).all() or ((scores < 0) | (scores > 1)).any():
        raise ValueError('Quality scores must be finite values in [0, 1]')
    studies = np.array([row['study'] for row in rows])
    regions = np.array([row['true_region'] for row in rows])
    overall = binary_metrics(quality, predicted, scores)
    overall['ci95_study_bootstrap'] = cluster_ci(quality, predicted, studies, scores, repeats)
    by_region = {}
    for region in ('spine', 'hip_left', 'hip_right', 'hip'):
        selected = regions == region if region != 'hip' else np.char.startswith(regions, 'hip_')
        metrics = binary_metrics(quality[selected], predicted[selected], scores[selected])
        metrics['ci95_study_bootstrap'] = cluster_ci(quality[selected], predicted[selected],
                                                     studies[selected], scores[selected], repeats)
        by_region[region] = metrics
    by_criterion = {}
    for criterion in CRITERIA:
        selected = np.array([labels[row['source_path']][criterion] in ('0', '1') for row in rows])
        indexed = np.flatnonzero(selected)
        y = np.array([int(labels[rows[i]['source_path']][criterion]) for i in indexed])
        official = VIOLATION_LABEL[criterion]
        pred = np.array([int(predicted[i] == 1 and official in rows[i]['violation_type'].split(';'))
                         for i in indexed])
        metrics = binary_metrics(y, pred)
        criterion_scores = []
        for i in indexed:
            raw = rows[i].get('criterion_scores')
            value = json.loads(raw).get(criterion) if raw else None
            criterion_scores.append(value)
        if criterion_scores and all(value is not None for value in criterion_scores):
            values = np.asarray(criterion_scores, dtype=float)
            if not np.isfinite(values).all() or ((values < 0) | (values > 1)).any():
                raise ValueError('Criterion scores must be finite values in [0, 1]')
            metrics = binary_metrics(y, pred, values)
            metrics['ci95_study_bootstrap'] = cluster_ci(y, pred, studies[indexed], values, repeats)
        else:
            metrics['ci95_study_bootstrap'] = cluster_ci(y, pred, studies[indexed], repeats=repeats)
        metrics['official_type'] = official
        by_criterion[criterion] = metrics
    macro_f1 = float(np.mean([item['f1'] for item in by_criterion.values()]))
    untyped = sum(row['quality_pred'] == '1' and not row['violation_type'].strip() for row in rows)
    wrong_class_type = sum(row['quality_pred'] == '0' and bool(row['violation_type'].strip()) for row in rows)
    return {
        'protocol': 'one-repeat study-held-out OOF; fixed decisions from saved inference path',
        'decision_versions': sorted({r.get('decision_version', 'legacy_unrecorded') for r in rows}),
        'labels_sha256': sha256(labels_path), 'predictions_sha256': sha256(oof_path),
        'labelled_images': len(rows), 'studies': len(fold_by_study), 'unlabelled_train_images_excluded': 3,
        'separate_test_dicom_without_labels': 3, 'bootstrap_resamples': repeats,
        'overall_quality': overall, 'by_region': by_region, 'by_criterion': by_criterion,
        'criterion_macro_f1': macro_f1, 'predicted_positive_without_type': untyped,
        'predicted_normal_with_type': wrong_class_type,
        'limitations': ['Internal cross-validation is not a closed or independent clinical test.',
                        'Saved OOF models differ from the already fitted shipped weights.',
                        'Type F1 uses final text decisions; type AUC is available only when matching criterion scores were saved.',
                        'No patient identifier proves patient-disjoint folds.'],
    }


def evaluate_release(labels_path: Path, results_path: Path) -> dict:
    """Diagnose shipped weights on labelled training images, one label per image."""
    with labels_path.open(newline='') as stream:
        labels = [row for row in csv.DictReader(stream) if row['quality_class'] in ('0', '1')]
    with results_path.open(newline='', encoding='utf-8-sig') as stream:
        results = {row['path_to_file']: row for row in csv.DictReader(stream)}
    if len(results) == 0 or any(row['first_source_path'] not in results for row in labels):
        raise ValueError('Release output does not cover every labelled image')
    paired = [(label, results[label['first_source_path']]) for label in labels]
    if any(result['processing_status'] != 'Success' for _, result in paired):
        raise ValueError('Release output failed to process a labelled image')
    quality = np.array([int(label['quality_class']) for label, _ in paired])
    predicted = np.array([int(result['quality_class']) for _, result in paired])
    scores = np.array([float(result['quality_prob']) for _, result in paired])
    if not np.isfinite(scores).all() or ((scores < 0) | (scores > 1)).any():
        raise ValueError('Release quality_prob is outside [0, 1]')
    regions = np.array([label['region'] for label, _ in paired])
    by_region = {}
    for region in ('spine', 'hip_left', 'hip_right', 'hip'):
        selected = regions == region if region != 'hip' else np.char.startswith(regions, 'hip_')
        by_region[region] = binary_metrics(quality[selected], predicted[selected], scores[selected])
    by_criterion = {}
    for criterion in CRITERIA:
        known = [(label, result) for label, result in paired if label[criterion] in ('0', '1')]
        y = np.array([int(label[criterion]) for label, _ in known])
        official = VIOLATION_LABEL[criterion]
        pred = np.array([int(result['quality_class'] == '1' and
                             official in result['violation_type'].split(';')) for _, result in known])
        by_criterion[criterion] = binary_metrics(y, pred)
    return {
        'protocol': 'apparent training-set diagnostic of already fitted shipped weights; not held-out accuracy',
        'results_sha256': sha256(results_path), 'labelled_unique_images': len(paired),
        'unlabelled_or_duplicate_full_batch_rows_excluded': len(results) - len(paired),
        'overall_quality': binary_metrics(quality, predicted, scores),
        'by_region': by_region, 'by_criterion': by_criterion,
        'criterion_macro_f1': float(np.mean([entry['f1'] for entry in by_criterion.values()])),
        'positive_without_official_type': sum(result['quality_class'] == '1' and not result['violation_type'].strip()
                                              for _, result in paired),
    }


def summarize_unlabelled_test(results_path: Path) -> dict:
    with results_path.open(newline='', encoding='utf-8-sig') as stream:
        rows = list(csv.DictReader(stream))
    validation_path = results_path.with_name('submission_validation.json')
    validation = json.loads(validation_path.read_text())
    return {
        'protocol': 'unlabelled organizer test DICOM; operational checks only',
        'results_sha256': sha256(results_path), 'files': len(rows),
        'success': sum(row['processing_status'] == 'Success' for row in rows),
        'failure': sum(row['processing_status'] == 'Failure' for row in rows),
        'predicted_positive': sum(row['processing_status'] == 'Success' and row['quality_class'] == '1'
                                  for row in rows),
        'predicted_positive_without_type': sum(row['processing_status'] == 'Success' and
                                               row['quality_class'] == '1' and
                                               not row['violation_type'].strip() for row in rows),
        'strict_table_valid': validation['valid'],
        'ground_truth_available': False,
        'accuracy_metrics': None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--labels', type=Path, default=Path('competition/labels/image_labels.csv'))
    parser.add_argument('--oof', type=Path, default=Path('competition/reports/scope_gate_1_9_pipeline_oof.csv'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--release-results', type=Path,
                        help='Optional full release CSV for a separately labelled training-set diagnostic')
    parser.add_argument('--unlabelled-test-results', type=Path,
                        help='Optional three-file organizer test output for operational checks')
    parser.add_argument('--bootstrap', type=int, default=1000)
    args = parser.parse_args()
    if args.bootstrap < 100:
        parser.error('--bootstrap must be at least 100')
    result = evaluate(args.labels, args.oof, args.bootstrap)
    if args.release_results:
        result['release_training_set_diagnostic'] = evaluate_release(args.labels, args.release_results)
    if args.unlabelled_test_results:
        result['unlabelled_test_operational'] = summarize_unlabelled_test(args.unlabelled_test_results)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({'output': str(args.output), 'overall': result['overall_quality'],
                      'criterion_macro_f1': result['criterion_macro_f1']}, ensure_ascii=False))


if __name__ == '__main__':
    main()
