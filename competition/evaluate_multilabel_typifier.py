"""Nested study-held-out evaluation. Release predictions are never overwritten."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import joblib
import pandas as pd

from dxaqc.model import group_of, official_violation_type
from evaluate_organizer_dataset import evaluate, sha256
from evaluate_structured_typifier import paired_f1_interval
from multilabel_typifier import MultilabelTypifier
from source_integrity import inspect_sources
from structured_typifier import target
from train import build_table, HERE


def run(dataset, labels_path, oof_path, output):
    if output.exists():
        raise ValueError('Choose a new experiment directory')
    audit = inspect_sources([('organiser', labels_path, dataset)])
    fingerprint = hashlib.sha256(json.dumps([(r['path'], r['pixel_sha256_current'])
                                             for r in audit['entries']]).encode()).hexdigest()
    labels = pd.read_csv(labels_path)
    labels = labels[labels.quality_class.notna()].reset_index(drop=True)
    features, embedding, _ = build_table(dataset, labels, HERE/'.cache', fingerprint)
    with labels_path.open() as stream:
        reference = {r['first_source_path']: r for r in csv.DictReader(stream) if r['quality_class'] in ('0','1')}
    with oof_path.open() as stream:
        original = list(csv.DictReader(stream))
    evaluate(labels_path, oof_path, repeats=100)
    index = {r.first_source_path: i for i, r in enumerate(labels.itertuples())}
    targets = [target(reference[p]) for p in labels.first_source_path]
    output.mkdir(parents=True)
    results, training_scope = {}, []
    for fold in sorted({r['fold'] for r in original}):
        held_out = {r['study'] for r in original if r['fold'] == fold}
        for group in ('spine', 'hip'):
            train = [i for i, r in enumerate(labels.itertuples()) if r.study_key not in held_out
                     and group_of(r.region) == group and targets[i] is not None]
            test = [r for r in original if r['fold'] == fold and group_of(r['predicted_region']) == group]
            testing = [index[r['source_path']] for r in test]
            model = MultilabelTypifier(group).fit([features[i] for i in train], embedding[train],
                                                [targets[i] for i in train], labels.iloc[train].study_key.tolist())
            for guard in (True, False):
                for row, prediction in zip(test, model.predict([features[i] for i in testing], embedding[testing], guard)):
                    results[row['source_path'], guard] = prediction
            training_scope.append({'fold': fold, 'group': group, 'training_images': len(train),
                                   'outer_test_studies': sorted(held_out), 'inner_splits': model.inner_splits,
                                   'thresholds': model.thresholds})
            joblib.dump(model, output/f'fold{fold}_{group}.joblib')
        print(f'Completed nested outer fold {fold}', flush=True)
    reports = {'released_v5': evaluate(labels_path, oof_path, repeats=1000)}
    for guard in (True, False):
        for policy in ('all_images', 'untyped_only'):
            rows = []
            for old in original:
                prediction = results[old['source_path'], guard]
                row = dict(old)
                row['typifier_evidence'] = json.dumps(prediction, ensure_ascii=False)
                if policy == 'all_images' or (old['quality_pred'] == '1' and not old['violation_type']):
                    row.update(quality_pred=str(int(bool(prediction['codes']))),
                               violation_type=official_violation_type(prediction['codes']),
                               criterion_scores=json.dumps(prediction['criterion_scores']),
                               decision_version='experimental-multilabel-1',
                               decision_reason='learned_criteria_or_normal')
                rows.append(row)
            name = ('guarded_' if guard else 'unguarded_')+policy
            path = output/(name+'_oof.csv')
            with path.open('w', newline='') as stream:
                writer = csv.DictWriter(stream, fieldnames=list(original[0])+['typifier_evidence'])
                writer.writeheader();writer.writerows(rows)
            reports[name] = evaluate(labels_path, path, repeats=1000)
            reports[name]['paired_delta_f1_ci95'] = paired_f1_interval(original, rows)
            reports[name]['axis_guard'] = guard
    for group in ('spine', 'hip'):
        selected = [i for i, r in enumerate(labels.itertuples()) if group_of(r.region)==group and targets[i] is not None]
        model = MultilabelTypifier(group).fit([features[i] for i in selected], embedding[selected],
                                            [targets[i] for i in selected], labels.iloc[selected].study_key.tolist())
        joblib.dump(model, output/f'multilabel_{group}.joblib')
    report = {'clinical_validation': False, 'release_modified': False,
              'protocol': 'Saved baseline outer study folds; three inner training-only study folds for criterion thresholds; fixed LR C=.01; no outer-test fitting',
              'excluded_incomplete_or_conflicting_targets': sum(t is None for t in targets),
              'training_scope': training_scope, 'policies': reports,
              'labels_sha256': sha256(labels_path), 'baseline_oof_sha256': sha256(oof_path),
              'code_sha256': {p: sha256(HERE/p) for p in ('multilabel_typifier.py','evaluate_multilabel_typifier.py')},
              'model_sha256': {p.name: sha256(p) for p in output.glob('*.joblib')},
              'limitations': ['Unguarded axis scores may conflict with the required physical angle and cannot be deployed.',
                              'Architecture evaluated on previously examined data; not independent post-selection validation.',
                              'A secondary normal verdict may increase missed binary alarms; untyped=0 alone is not an improvement.',
                              'Quality score retained from baseline to isolate the change in categorical decisions; no new probability calibration claimed.']}
    (output/'evaluation.json').write_text(json.dumps(report, indent=2, ensure_ascii=False)+'\n')
    print(json.dumps({name: {'f1': r['overall_quality']['f1'], 'sensitivity': r['overall_quality']['sensitivity'],
                            'type_f1':r['criterion_macro_f1'],'untyped':r['predicted_positive_without_type']}
                      for name,r in reports.items()},indent=2))


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('dataset','labels','oof','output'):
        p.add_argument('--'+name,required=True,type=Path)
    a=p.parse_args()
    run(a.dataset,a.labels,a.oof,a.output)
