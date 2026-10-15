"""Study-exclusive global quality head sharing examples across spine and hip.

Typed decisions are retained from the supplied candidate. Only quality ranking
is replaced or blended. Mixing weights use inner OOF AUC; external targets are
used only by the evaluator. This tests sample sharing, not another encoder.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from dxaqc.model import _matrix
from evaluate_organizer_dataset import evaluate, sha256
from experiments.three_iteration_upgrade import fit_family, predict_family, fit_decoder, decode_scores, dump
from experiments.verify_metric_candidate import compare
from gpu_research.common import load_feature_bundle

EXECUTION_SOURCE = Path(__file__).read_text()
EXECUTION_SHA256 = sha256(Path(__file__))


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
        if (sha256(p / 'bank.joblib') != manifest['bank_sha256'] or manifest['labels_sha256'] != sha256(args.labels) or
                manifest['baseline_sha256'] != sha256(args.baseline)):
            raise ValueError('Head bank or target identity differs')
    if manifests[0]['source_fingerprint'] != manifests[1]['source_fingerprint']:
        raise ValueError('Source identities differ')
    bank, prior = [joblib.load(p / 'bank.joblib') for p in (args.bank, args.ranking_bank)]
    labels = pd.read_csv(args.labels)
    labels = labels[labels.quality_class.notna()].reset_index(drop=True)
    candidate = pd.read_csv(args.candidate).sort_values('index').reset_index(drop=True)
    identity = ['index', 'fold', 'study', 'source_path', 'quality_true']
    if not candidate[identity].equals(bank['baseline'].sort_values('index').reset_index(drop=True)[identity]):
        raise ValueError('Candidate evaluation identities differ')
    columns = sorted({k for row in bank['features'] for k in row})
    geometry = _matrix(bank['features'], columns)
    embeddings = [load_feature_bundle(Path(e['path']), labels, manifests[0]['source_fingerprint'], sha256(args.labels))[0]
                  for e in manifests[0]['embeddings']]
    target = labels.quality_class.to_numpy(float)
    groups = labels.study_key.astype(str).to_numpy()
    tables = {name: candidate.copy() for name in ('global_only', 'fixed_half', 'inner_auc_blend')}
    saved, audit = [], []
    args.output.mkdir(parents=True)
    (args.output / 'executed_global_quality_head.py').write_text(EXECUTION_SOURCE)
    with threadpool_limits(limits=4):
        for fold in sorted(candidate.fold.unique()):
            test = np.flatnonzero(candidate.fold.to_numpy() == fold)
            held = set(groups[test])
            train = np.flatnonzero(~np.isin(groups, list(held)))
            inner = np.full(len(train), np.nan)
            for it, iv in GroupKFold(4).split(train, groups=groups[train]):
                heads = fit_family(geometry, embeddings, target, train[it])
                inner[iv] = predict_family(heads, geometry, embeddings, train[iv])
            heads = fit_family(geometry, embeddings, target, train)
            outer = predict_family(heads, geometry, embeddings, test)
            prior_inner, prior_outer = np.full(len(train), np.nan), np.full(len(test), np.nan)
            for state in prior['folds']:
                if state['fold'] != fold:
                    continue
                if set(groups[state['train']]) & held or not set(state['train']) <= set(train):
                    raise ValueError('Prior model includes held-out studies')
                spec = fit_decoder(state, 1)
                _, train_score = decode_scores(spec, state['codes'], np.column_stack([state['inner'][c] for c in state['codes']]), state['inner']['quality'])
                _, test_score = decode_scores(spec, state['codes'], np.column_stack([state['outer'][c] for c in state['codes']]), state['outer']['quality'])
                prior_inner[np.searchsorted(train, state['train'])] = train_score
                prior_outer[np.searchsorted(test, state['test'])] = test_score
            if not np.isfinite(np.r_[inner, outer, prior_inner, prior_outer]).all():
                raise ValueError('Incomplete study-exclusive predictions')
            weight = max((0., .25, .5, .75, 1.), key=lambda a: (
                roc_auc_score(target[train], a * inner + (1 - a) * prior_inner), -abs(a - .5)))
            for name, a in [('global_only', 1.), ('fixed_half', .5), ('inner_auc_blend', weight)]:
                tables[name].loc[test, 'quality_score'] = a * outer + (1 - a) * prior_outer
                tables[name].loc[test, 'decision_version'] = f'research-global-ranking-{name}'
            saved.append({'fold': int(fold), 'models': heads, 'columns': columns, 'selected_weight': weight})
            audit.append({'fold': int(fold), 'training_studies': len(set(groups[train])),
                          'held_out_studies': len(held), 'selected_global_weight': weight})
            print(f'Global QC fold {fold} complete', flush=True)
    joblib.dump(saved, args.output / 'global_heads.joblib')
    for name, table in tables.items():
        directory = args.output / name
        directory.mkdir()
        path = directory / 'candidate_oof.csv'
        table.to_csv(path, index=False)
        metrics = evaluate(args.labels, path, repeats=2000)
        screen = compare(args.baseline, path, args.labels)
        dump(directory / 'evaluation.json', {'protocol': __doc__, 'metrics': metrics, 'screen': screen,
             'selection_audit': audit, 'bank_sha256': manifests[0]['bank_sha256'],
             'ranking_bank_sha256': manifests[1]['bank_sha256'], 'code_sha256': EXECUTION_SHA256,
             'clinical_validation': False, 'automatic_promotions': 0, 'typed_decisions_unchanged': True})
        print(json.dumps({'variant': name, 'quality': metrics['overall_quality']}), flush=True)


if __name__ == '__main__':
    main()
