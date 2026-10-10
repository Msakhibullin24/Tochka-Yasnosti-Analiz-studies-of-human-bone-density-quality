"""E21: select head regularisation inside the training studies, by AUC rather than by F1.

Every head in this project has run on fixed hyperparameters since the first screen. That is one
untouched degree of freedom, and the pilot says it is not flat: mean inner out-of-fold AUC for
the logistic head moves from 0.901 to 0.921 across the C grid on spine_coverage, and from
0.850 down to 0.718 on spine_artifact, where four folds out of five agree on the direction.

The pilot also says how not to do this. The argmax C per outer fold is unstable, picking 5.0,
0.2, 1.0, 0.2 and 0.05 across the five folds of one criterion, while the whole AUC spread
across the grid is only 0.02 to 0.03 inside a fold. Selection would chase noise. Two choices
follow.

The criterion for selection is mean inner out-of-fold AUC, never inner out-of-fold F1. This
project has already measured that inner F1 misranks candidates on 23 of 25 criterion-folds,
which is also why the rare criteria score F1 0.17 to 0.46 while separating at AUC 0.84 to
0.94: the cut cannot be placed, and that is a threshold problem wearing a selection problem's
clothes.

The search happens entirely inside the training studies of each outer fold, under an inner
GroupKFold on study, so no study straddles either level. The outer test fold is read once, for
scoring.

Two things had to be right before any of this measured the intended thing, and both were got
wrong first, so the guards are in the code.

The score is the mean rank of the geometry heads AND the frozen encoder heads, exactly as the
shipped pipeline builds it. A geometry-only version of this experiment scored 0.45 against a
member baseline of 0.61, not because tuning hurts but because it had quietly deleted half the
signal. The bundle argument is therefore mandatory rather than optional.

Ranks are taken within the anatomy-matched rows. Ranking over all 249 rows lifts the other
region's NaN rows to the top of the scale, compresses the real scores into the bottom,
collapses the threshold and flags every row, which looked like a catastrophic result until it
was traced to arithmetic.

A --fixed arm runs this same code with untuned hyperparameters. If it does not reproduce the
published member F1, the tuned arm is measuring something else and the script refuses to
compare them.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from threadpoolctl import threadpool_limits

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
from dxaqc.decision import decide  # noqa: E402
from dxaqc.model import CRITERIA, group_of, official_violation_type  # noqa: E402
from evaluate_organizer_dataset import evaluate  # noqa: E402
from experiments.seed_council import bagged_threshold  # noqa: E402
from experiments.verify_metric_candidate import compare  # noqa: E402
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis  # noqa: E402
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402
from sklearn.model_selection import GroupKFold  # noqa: E402
from sklearn.pipeline import Pipeline, make_pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402
from sklearn.svm import SVC  # noqa: E402

# Three values per head. Wider buys inner score at the cost of outer score, and the middle
# entry of each list is the value the shipped pipeline already uses, so the --fixed arm is
# exactly the shipped model.
GRID = {
    'rf': [{'min_samples_leaf': 1}, {'min_samples_leaf': 2}, {'min_samples_leaf': 8}],
    'gb': [{'learning_rate': 0.06, 'max_leaf_nodes': 15},
           {'learning_rate': 0.03, 'max_leaf_nodes': 7},
           {'learning_rate': 0.12, 'max_leaf_nodes': 31}],
    'lr': [{'C': 0.005}, {'C': 0.05}, {'C': 5.0}],
    'svc': [{'C': 0.1}, {'C': 1.0}, {'C': 10.0}],
    'lda': [{'shrinkage': 'auto'}, {'shrinkage': 0.2}, {'shrinkage': 0.8}],
}
EMB_GRID = [{'C': 0.005}, {'C': 0.05}, {'C': 5.0}]
MEMBERS = ('rf', 'gb', 'lr', 'svc', 'lda')
INNER_FOLDS = 3


def build(name, params, seed):
    if name == 'rf':
        return make_pipeline(StandardScaler(),
                             ExtraTreesClassifier(n_estimators=600, class_weight='balanced_subsample',
                                                  random_state=seed, n_jobs=1, **params))
    if name == 'gb':
        return HistGradientBoostingClassifier(max_iter=300, min_samples_leaf=8, l2_regularization=1.,
                                              class_weight='balanced', random_state=seed, **params)
    if name == 'lr':
        return make_pipeline(StandardScaler(),
                             LogisticRegression(class_weight='balanced', max_iter=5000,
                                                random_state=seed, **params))
    if name == 'svc':
        return make_pipeline(StandardScaler(),
                             SVC(gamma='scale', probability=True, random_state=seed, **params))
    return make_pipeline(StandardScaler(),
                         LinearDiscriminantAnalysis(solver='lsqr', **params))


def classes_of(model):
    return model[-1].classes_ if isinstance(model, Pipeline) else model.classes_


def inner_oof_and_test(make, X_train, y_train, g_train, X_test, seed, folds=INNER_FOLDS):
    """Out-of-fold scores on the training studies plus fitted predictions on the test rows.

    The operating point is chosen from the first part only, which is what keeps the outer test
    fold out of the threshold.
    """
    oof = np.full(len(y_train), np.nan)
    if len(np.unique(y_train)) >= 2 and len(y_train) >= 2 * folds:
        for tr, te in GroupKFold(n_splits=folds).split(X_train, y_train, g_train):
            if len(np.unique(y_train[tr])) < 2:
                continue
            m = make(seed)
            m.fit(X_train[tr], y_train[tr])
            cl = classes_of(m)
            if 1 in cl:
                oof[te] = m.predict_proba(X_train[te])[:, list(cl).index(1)]
    m = make(seed)
    m.fit(X_train, y_train)
    cl = classes_of(m)
    pred = m.predict_proba(X_test)[:, list(cl).index(1)] if 1 in cl else np.zeros(len(X_test))
    return oof, pred


def select(name, make_grid, X, y, g, seed):
    """Pick by mean inner out-of-fold AUC on the training studies only."""
    default = GRID[name][1] if name in GRID else EMB_GRID[1]
    if len(np.unique(y)) < 2 or len(y) < 4 * INNER_FOLDS:
        return default
    best, best_auc = default, -np.inf
    for params in make_grid:
        oof, _ = inner_oof_and_test(lambda sd, p=params: build(name, p, sd), X, y, g, X[:1], seed)
        ok = np.isfinite(oof)
        if ok.sum() < 4 or len(np.unique(y[ok])) < 2:
            continue
        auc = float(roc_auc_score(y[ok], oof[ok]))
        if auc > best_auc:
            best_auc, best = auc, params
    return best


def rank_within(values, n):
    """Rank the finite entries into [0, 1] and leave the rest NaN."""
    out = np.full(n, np.nan)
    present = np.isfinite(values)
    if present.sum() > 1:
        out[present] = np.argsort(np.argsort(values[present])) / (present.sum() - 1)
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--labels', type=Path, default=HERE / 'labels/image_labels.csv')
    parser.add_argument('--baseline', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--bundle', action='append', required=True, metavar='NAME=PATH.npy')
    parser.add_argument('--bags', type=int, default=200)
    parser.add_argument('--seeds', type=int, nargs='+', default=[17, 101, 202, 303])
    parser.add_argument('--fixed', action='store_true',
                        help='Control arm: untuned hyperparameters through this same code, which '
                             'must reproduce the published member F1 for the tuned arm to mean anything.')
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('Choose a new output directory')
    args.output.mkdir(parents=True)
    torch.set_num_threads(4)

    from train import build_table
    from source_integrity import inspect_sources
    from gpu_research.common import load_feature_bundle

    labels = pd.read_csv(args.labels)
    labels = labels[labels.quality_class.notna()].reset_index(drop=True)
    baseline = pd.read_csv(args.baseline).sort_values('index').reset_index(drop=True)
    audit = inspect_sources([('organiser', args.labels, args.dataset)])
    fingerprint = hashlib.sha256(json.dumps(
        [(r['path'], r['pixel_sha256_current']) for r in audit['entries']]).encode()).hexdigest()
    features = [dict(f) for f in build_table(args.dataset, labels, HERE / '.cache', fingerprint)[0]]
    for row in baseline.itertuples():
        if row.predicted_region == 'spine':
            features[row.index]['spine_abs_angle_deg'] = json.loads(
                row.criterion_states).get('spine_axis', {}).get('angle_deg')

    labels_sha = hashlib.sha256(args.labels.read_bytes()).hexdigest()
    bundles, bundle_paths = {}, {}
    for item in args.bundle:
        name, sep, path = item.partition('=')
        if not sep or not path:
            parser.error(f'--bundle expects NAME=PATH.npy, got {item!r}')
        bundle_paths[name] = path
        bundles[name] = load_feature_bundle(Path(path), labels, fingerprint, labels_sha)[0]
        print(f'bundle {name}: {bundles[name].shape}', flush=True)

    keys = sorted({k for f in features for k in f})
    X_geom = np.array([[f.get(k, np.nan) for k in keys] for f in features], float)
    groups = labels.study_key.astype(str).to_numpy()
    anatomy = np.array([group_of(r) for r in labels.region])
    studies = baseline['study'].astype(str).to_numpy()
    folds_col = baseline['fold'].to_numpy()
    n = len(baseline)

    def matrix_for(idx):
        sub = X_geom[idx]
        fill = np.nanmedian(sub, axis=0)
        fill = np.where(np.isfinite(fill), fill, 0.0)
        return np.where(np.isfinite(sub), sub, fill[None, :])

    def build_score(tr, te, criterion, seed, fold, log):
        target = pd.to_numeric(labels[criterion], errors='coerce').to_numpy(float)
        known = np.isfinite(target[tr])
        if known.sum() < 5 or len(np.unique(target[tr][known].astype(int))) < 2:
            return None, float(np.nextafter(1., np.inf))
        truth = target[tr][known].astype(int)
        Xtr = matrix_for(tr[known])
        gtr = groups[tr[known]]
        arms = []
        for name in MEMBERS:
            params = GRID[name][1] if args.fixed else select(
                name, GRID[name], Xtr, truth, gtr, seed)
            if not args.fixed:
                log.append({'seed': seed, 'fold': int(fold), 'criterion': criterion,
                        'head': name, 'params': params})
            oof, pred = inner_oof_and_test(lambda sd, p=params: build(name, p, sd),
                                           Xtr, truth, gtr, matrix_for(te), seed)
            values = np.full(n, np.nan)
            values[tr[known]] = oof
            values[te] = pred
            arms.append(rank_within(values, n))
        geom = np.nanmean(np.column_stack(arms), axis=1)  # mean over heads, one value per row
        for bname, bmat in bundles.items():
            params = EMB_GRID[1] if args.fixed else select(
                'lr', EMB_GRID, bmat[tr[known]], truth, gtr, seed)
            if not args.fixed:
                log.append({'seed': seed, 'fold': int(fold), 'criterion': criterion,
                            'head': f'lr@{bname}', 'params': params})
            oof, pred = inner_oof_and_test(lambda sd, p=params: build('lr', p, sd),
                                           bmat[tr[known]], truth, gtr, bmat[te], seed)
            values = np.full(n, np.nan)
            values[tr[known]] = oof
            values[te] = pred
            arms.append(rank_within(values, n))
        # axis=1 averages over the heads and keeps one value per row. axis=0 would average
        # over images and return one value per head, which is a (6,) vector; that was the
        # bug, and it surfaced as a 6-element "score column".
        column = np.nanmean(np.column_stack(arms), axis=1)
        if np.ndim(column) != 1 or column.shape[0] != n:
            raise ValueError(f'column shape {column.shape}, arms {[a.shape for a in arms]}, '
                             f'n={n}, tr={tr.shape}, te={te.shape}, known.sum={known.sum()}')
        scores = column[tr[known]]
        if len(scores) != len(truth) or len(column) != n:
            raise ValueError(f'shape mismatch: column={column.shape} scores={scores.shape} '
                             f'truth={truth.shape} n={n} tr={tr.shape} known={known.shape} '
                             f'arms={[a.shape for a in arms]}')
        thr = bagged_threshold(truth, scores, args.bags,
                               seed + 100 + 7 * int(fold))
        return column, thr

    selection_log = []
    variants = {}
    for seed in args.seeds:
        table = baseline.copy()
        for fold in sorted(baseline.fold.unique()):
            held = set(studies[folds_col == fold])
            tr_all = np.flatnonzero(~np.isin(studies, list(held)))
            te_all = np.flatnonzero(np.isin(studies, list(held)))
            for group in ('spine', 'hip'):
                tr = tr_all[anatomy[tr_all] == group]
                te = te_all[anatomy[te_all] == group]
                if not len(te):
                    continue
                accumulated, thresholds = {}, {}
                for criterion in CRITERIA[group]:
                    column, thr = build_score(tr, te, criterion, seed, fold, selection_log)
                    thresholds[criterion] = thr
                    accumulated[criterion] = (np.zeros(len(te)) if column is None
                                              else column[te])
                for position, row_index in enumerate(te):
                    criterion_scores = {k: float(v[position]) for k, v in accumulated.items()}
                    probability = float(1 - np.prod([1 - p for p in criterion_scores.values()]))
                    decision = decide(group, probability, criterion_scores, features[row_index],
                                      float(np.nextafter(1., np.inf)), thresholds)
                    for key, value in {'quality_score': probability,
                                       'quality_pred': decision['quality'],
                                       'violation_type': official_violation_type(decision['violations']),
                                       'criterion_states': json.dumps(decision['criterion_states'], sort_keys=True),
                                       'criterion_scores': json.dumps(criterion_scores, sort_keys=True),
                                       'criterion_thresholds': json.dumps(thresholds, sort_keys=True),
                                       'quality_threshold': decision['quality_threshold'],
                                       'decision_version': 'research-head-tuning-1',
                                       'decision_reason': decision['decision_reason']}.items():
                        table.at[row_index, key] = value
        name = f'seed{seed}'
        path = args.output / f'{name}_oof.csv'
        table.to_csv(path, index=False)
        variants[name] = {'metrics': evaluate(args.labels, path, repeats=2000)}
        print(f'{name}: F1={variants[name]["metrics"]["overall_quality"]["f1"]:.4f}', flush=True)

    seed_f1 = [variants[f'seed{s}']['metrics']['overall_quality']['f1'] for s in args.seeds]
    ref = json.loads(Path('data/gpu-runs/council-bagged/evaluation.json').read_text())
    ref_members = {k: v['metrics']['overall_quality']['f1']
                   for k, v in ref['variants'].items() if k != 'council'}
    report = {
        'protocol': __doc__, 'fixed_control_arm': bool(args.fixed), 'grid': GRID,
        'emb_grid': EMB_GRID, 'inner_folds': INNER_FOLDS,
        'selection_criterion': 'mean inner out-of-fold AUC on training studies only',
        'bundles': {name: str(path) for name, path in bundle_paths.items()},
        'variants': variants,
        'seed_mean_f1': float(np.mean(seed_f1)), 'seed_sd_f1': float(np.std(seed_f1, ddof=1)),
        'published_member_f1': ref_members,
        'published_member_mean_f1': ref['member_mean_f1'],
        'published_member_sd_f1': ref['member_sd_f1'],
        'published_council_f1': ref['variants']['council']['metrics']['overall_quality']['f1'],
        'selection_audit': selection_log,
        'clinical_validation': False, 'model_affects_decision': False,
        'limitations': ['Three grid points per head by design.',
                        'The argmax was unstable across outer folds in the pilot, so part of any gain is '
                        'expected to be selection noise; the seed spread is reported alongside.',
                        'No automatic promotion; the frozen screen decides.']}
    (args.output / 'evaluation.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n',
                                                encoding='utf-8')
    print(f'\narm={"fixed" if args.fixed else "tuned"}  seed mean F1 {np.mean(seed_f1):.4f} '
          f'(sd {np.std(seed_f1, ddof=1):.4f})')
    print(f'published member mean F1 {ref["member_mean_f1"]:.4f} '
          f'(sd {ref["member_sd_f1"]:.4f}), council {report["published_council_f1"]:.4f}')


if __name__ == '__main__':
    with threadpool_limits(limits=4):
        main()