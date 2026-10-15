"""Nested study-exclusive nonlinear quality ranking over frozen encoders.

Kernel/C and blend weight selected inside each outer training split. Typed
classes stay supplied candidate's classes. Repeated cohort use is exploratory.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, normalize
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.svm import SVC
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from dxaqc.model import _matrix
from evaluate_organizer_dataset import evaluate, sha256
from experiments.three_iteration_upgrade import fit_decoder, decode_scores, dump
from experiments.verify_metric_candidate import compare
from gpu_research.common import load_feature_bundle

SOURCE = Path(__file__).read_bytes()
SOURCE_SHA256 = hashlib.sha256(SOURCE).hexdigest()
CONFIGURATIONS = [('linear', c, 1.) for c in (.01, .1, 1., 10.)] + [('rbf', c, gamma) for c in (.1, 1., 10.) for gamma in (.1, 1., 10.)]


def fit_head(geometry, embeddings, target, ids, configuration):
    kernel, regularization, gamma = configuration
    imputer = SimpleImputer(strategy='median', keep_empty_features=True)
    scaler = StandardScaler()
    geo = np.clip(scaler.fit_transform(imputer.fit_transform(geometry[ids])), -8, 8)
    geo *= .25 / np.sqrt(geo.shape[1])
    x = np.column_stack([*(normalize(v[ids]) / np.sqrt(len(embeddings)) for v in embeddings), geo])
    head = SVC(C=regularization, kernel=kernel, gamma=gamma, class_weight='balanced')
    head.fit(x, target[ids])
    return {'imputer': imputer, 'scaler': scaler, 'head': head}


def predict_head(model, geometry, embeddings, ids):
    geo = np.clip(model['scaler'].transform(model['imputer'].transform(geometry[ids])), -8, 8)
    geo *= .25 / np.sqrt(geo.shape[1])
    x = np.column_stack([*(normalize(v[ids]) / np.sqrt(len(embeddings)) for v in embeddings), geo])
    return model['head'].decision_function(x)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('bank', 'ranking_bank', 'candidate', 'output'):
        parser.add_argument('--' + name.replace('_', '-'), type=Path, required=True)
    parser.add_argument('--labels', type=Path, default=ROOT / 'labels/image_labels.csv')
    parser.add_argument('--baseline', type=Path, default=ROOT / 'reports/quality_review_axis_guard_posthoc_2026_09_29.csv')
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('Choose a new output directory')
    manifests = [json.loads((p / 'manifest.json').read_text()) for p in (args.bank, args.ranking_bank)]
    for p, manifest in zip((args.bank, args.ranking_bank), manifests):
        if (sha256(p / 'bank.joblib') != manifest['bank_sha256'] or manifest['labels_sha256'] != sha256(args.labels)
                or manifest['baseline_sha256'] != sha256(args.baseline)):
            raise ValueError('Head bank or target identity differs')
    if manifests[0]['source_fingerprint'] != manifests[1]['source_fingerprint']:
        raise ValueError('Source identities differ')
    bank, reference = [joblib.load(p / 'bank.joblib') for p in (args.bank, args.ranking_bank)]
    prior = {(s['fold'], s['group']): s for s in reference['folds']}
    labels = pd.read_csv(args.labels)
    labels = labels[labels.quality_class.notna()].reset_index(drop=True)
    candidate = pd.read_csv(args.candidate).sort_values('index').reset_index(drop=True)
    identity = ['index', 'fold', 'study', 'source_path', 'quality_true']
    if not candidate[identity].equals(bank['baseline'].sort_values('index').reset_index(drop=True)[identity]):
        raise ValueError('Candidate identities differ')
    columns = sorted({k for row in bank['features'] for k in row})
    geometry = _matrix(bank['features'], columns)
    embeddings = [load_feature_bundle(Path(e['path']), labels, manifests[0]['source_fingerprint'], sha256(args.labels))[0]
                  for e in manifests[0]['embeddings']]
    target = labels.quality_class.to_numpy(int)
    groups = labels.study_key.astype(str).to_numpy()
    tables = {name: candidate.copy() for name in ('nonlinear_only', 'fixed_half', 'inner_auc_blend')}
    saved, audit = [], []
    args.output.mkdir(parents=True)
    (args.output / 'executed_nonlinear_quality_head.py').write_bytes(SOURCE)
    with threadpool_limits(limits=4):
        for state in bank['folds']:
            train, test = state['train'], state['test']
            if set(groups[train]) & set(groups[test]):
                raise ValueError('Outer studies overlap')
            inner_candidates = []
            splits = list(GroupKFold(4).split(train, groups=groups[train]))
            for configuration in CONFIGURATIONS:
                inner = np.full(len(train), np.nan)
                for it, iv in splits:
                    model = fit_head(geometry, embeddings, target, train[it], configuration)
                    inner[iv] = predict_head(model, geometry, embeddings, train[iv])
                inner_candidates.append(inner)
            best = max(range(len(CONFIGURATIONS)), key=lambda i: (roc_auc_score(target[train], inner_candidates[i]), -i))
            raw_inner = inner_candidates[best]
            calibrator = LogisticRegression(C=1., max_iter=2000).fit(raw_inner[:, None], target[train])
            inner = calibrator.predict_proba(raw_inner[:, None])[:, 1]
            model = fit_head(geometry, embeddings, target, train, CONFIGURATIONS[best])
            outer = calibrator.predict_proba(predict_head(model, geometry, embeddings, test)[:, None])[:, 1]
            old = prior[state['fold'], state['group']]
            if not np.array_equal(train, old['train']) or not np.array_equal(test, old['test']):
                raise ValueError('Reference training/held-out fold membership differs')
            spec = fit_decoder(old, 1)
            _, prior_inner = decode_scores(spec, old['codes'], np.column_stack([old['inner'][c] for c in old['codes']]), old['inner']['quality'])
            _, prior_outer = decode_scores(spec, old['codes'], np.column_stack([old['outer'][c] for c in old['codes']]), old['outer']['quality'])
            weight = max((0., .25, .5, .75, 1.), key=lambda a: (roc_auc_score(target[train], a * inner + (1 - a) * prior_inner), -a))
            for name, a in [('nonlinear_only', 1.), ('fixed_half', .5), ('inner_auc_blend', weight)]:
                tables[name].loc[test, 'quality_score'] = a * outer + (1 - a) * prior_outer
                tables[name].loc[test, 'decision_version'] = f'research-nonlinear-ranking-{name}'
            saved.append({'fold': state['fold'], 'group': state['group'], 'model': model,
                          'calibrator': calibrator, 'columns': columns, 'selected_weight': weight})
            audit.append({'fold': state['fold'], 'group': state['group'], 'configuration': CONFIGURATIONS[best],
                          'inner_auc': float(roc_auc_score(target[train], raw_inner)), 'selected_weight': weight,
                          'outer_studies_used_for_selection': False})
            print(json.dumps(audit[-1]), flush=True)
    joblib.dump(saved, args.output / 'nonlinear_heads.joblib')
    for name, table in tables.items():
        directory = args.output / name
        directory.mkdir()
        path = directory / 'candidate_oof.csv'
        table.to_csv(path, index=False)
        metrics = evaluate(args.labels, path, repeats=2000)
        screen = compare(args.baseline, path, args.labels)
        dump(directory / 'evaluation.json', {'protocol': __doc__, 'metrics': metrics, 'screen': screen,
             'selection_audit': audit, 'bank_sha256': manifests[0]['bank_sha256'],
             'ranking_bank_sha256': manifests[1]['bank_sha256'], 'code_sha256': SOURCE_SHA256,
             'clinical_validation': False, 'automatic_promotions': 0, 'typed_decisions_unchanged': True})
        print(json.dumps({'name': name, 'quality': metrics['overall_quality']}), flush=True)


if __name__ == '__main__':
    main()
