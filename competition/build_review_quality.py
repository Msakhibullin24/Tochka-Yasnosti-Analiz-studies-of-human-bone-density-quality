"""Freeze supervised review of previously untyped alarms; report actual score changes."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from dxaqc.joint_quality import JointQualityModel
from dxaqc.model import group_of, official_violation_type
from evaluate_organizer_dataset import evaluate, sha256
from source_integrity import inspect_sources
from structured_typifier import target
from train import build_table, HERE


def run(args):
    if args.output.exists():
        raise ValueError('Choose a new experiment directory')
    evaluate(args.labels, args.baseline, repeats=20)
    with args.baseline.open(newline='') as f:
        rows = list(csv.DictReader(f))
    with args.secondary.open(newline='') as f:
        secondary = list(csv.DictReader(f))
    indexed = {r['source_path']: r for r in secondary}
    if len(indexed) != len(secondary) or {r['source_path'] for r in rows} != set(indexed):
        raise ValueError('Mismatched or duplicate secondary identities')
    changed = 0
    for row in rows:
        source = indexed[row['source_path']]
        if any(source[k] != row[k] for k in ('fold', 'study', 'quality_true', 'predicted_region')):
            raise ValueError('Secondary fold or identity mismatch')
        if row['quality_pred'] == '1' and not row['violation_type']:
            prediction = json.loads(source['typifier_evidence'])
            row['quality_pred'] = str(int(bool(prediction['codes'])))
            row['quality_score'] = str(1.-prediction['normal_score'])
            row['violation_type'] = official_violation_type(prediction['codes'])
            row['decision_version'] = 'review-1'
            row['decision_reason'] = 'supervised_type_review' if prediction['codes'] else 'supervised_normal_review'
            states = json.loads(row['criterion_states'])
            for code in prediction['codes']:
                states[code].update(status='fail', basis='supervised_untyped_review', clinical_validation=False)
            row['criterion_states'] = json.dumps(states, sort_keys=True)
            row['criterion_scores'] = ''
            changed += 1
    args.output.mkdir(parents=True)
    path = args.output/'review_oof.csv'
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    metrics = evaluate(args.labels, path, repeats=200)
    audit = inspect_sources([('organiser', args.labels, args.dataset)])
    fingerprint = hashlib.sha256(json.dumps([(e['path'], e['pixel_sha256_current']) for e in audit['entries']]).encode()).hexdigest()
    labels = pd.read_csv(args.labels); labels = labels[labels.quality_class.notna()].reset_index(drop=True)
    features, embeddings, _ = build_table(args.dataset, labels, HERE/'.cache', fingerprint)
    with args.labels.open(newline='') as f:
        references = {r['first_source_path']: r for r in csv.DictReader(f) if r['quality_class'] in ('0', '1')}
    targets = [target(references[p]) for p in labels.first_source_path]
    models = {}
    for group in ('spine', 'hip'):
        indices = [i for i, r in enumerate(labels.itertuples()) if group_of(r.region) == group and targets[i] is not None]
        models[group] = JointQualityModel(group, 'advisory').fit([features[i] for i in indices], embeddings[indices], [targets[i] for i in indices])
    bundle = {'schema_version': 1, 'mode': 'review_untyped', 'models': models,
              'labels_sha256': sha256(args.labels), 'clinical_validation': False}
    joblib.dump(bundle, args.output/'quality_review.joblib')
    report = {'protocol': 'Frozen baseline outer study folds and corresponding supervised secondary predictions; fixed argmax; no new labels',
              'metrics': metrics, 'reviewed_oof_alarms': changed, 'mode': 'review_untyped',
              'labels_sha256': sha256(args.labels), 'secondary_sha256': sha256(args.secondary),
              'model_sha256': sha256(args.output/'quality_review.joblib'), 'code_sha256': sha256(Path(__file__)),
              'clinical_validation': False,
              'limitations': ['Previously examined OOF, not a new independent test.',
                              'Sensitivity tradeoff must be reported alongside F1 and type F1.',
                              'Axis label is learned from expert criterion labels; geometric disagreement remains explicit.',
                              'Final fitted models are different from the fold-specific instances.']}
    (args.output/'evaluation.json').write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps({k: metrics['overall_quality'][k] for k in ('f1', 'sensitivity', 'roc_auc')}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('dataset', 'labels', 'baseline', 'secondary', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    args = parser.parse_args(); run(args)
