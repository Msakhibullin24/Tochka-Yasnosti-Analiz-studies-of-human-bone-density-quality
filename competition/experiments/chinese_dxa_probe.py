"""DXA representation probe with held-out conservative filename groups.

This is the cheapest test of the whole in-domain premise, and it is deliberately run before
any self-supervised training. The argument for pretraining on fourteen thousand DXA images is
that they are the right modality. If that is true, frozen features from those images should
predict something clinically real about the same patients, on patients the probe never saw. If
they cannot, then the representation does not encode DXA-relevant structure and spending GPU
days on masked image modelling over it is buying nothing.

The target is bone mineral density or T-score from the corpus workbook. It is a different
quantity from any QC criterion, which is exactly the point: it is abundant, it is real, and it
shares no labels with the 249 competition frames, so nothing here can leak into the held-out
competition folds or be tuned on them.

Splits group the same filename name token across dates and case differences.
Patient identities are not independently verified. A demographics-only model
is fitted alongside. Ridge regularization and imputation are fitted inside the
outer training groups. The historical version selected alpha on external OOF
and misaligned rows after filtering; its metrics require recomputation.

No threshold is tuned here. The result is a comparison against a fixed baseline, and the
decision it feeds is whether to attempt self-supervised pretraining at all.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
from gpu_research.common import dump  # noqa: E402
from experiments.prepare_chinese_dxa import person_group

BMD_RE = re.compile(r'(t[-_ ]?score|bmd|density|g/cm|骨密度)', re.I)
DEMO_RE = re.compile(r'^(age|sex|gender|height|weight|bmi|年龄|性别|身高|体重)', re.I)


def load_workbook(path: Path) -> dict[str, pd.DataFrame]:
    sheets = pd.read_excel(path, sheet_name=None)
    return {k: v for k, v in sheets.items() if v is not None and len(v.columns)}


def find_targets(sheets: dict[str, pd.DataFrame], meta: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Attach workbook rows to patients by the pinyin name in the filename."""
    patients = sorted(meta.patient.unique())
    names = {}
    for p in patients:
        # A workbook name without date cannot resolve multiple acquisitions.
        names.setdefault(person_group(p), []).append(p)
    names = {name: keys[0] for name, keys in names.items() if len(keys) == 1}
    frames, used = [], []
    for sheet, df in sheets.items():
        cols = {c: str(c) for c in df.columns}
        key = next((c for c in cols if any(k in cols[c].lower() for k in ('name', 'patient', '姓名'))), None)
        if key is None:
            continue
        tcol = next((c for c in cols if BMD_RE.search(cols[c])), None)
        if tcol is None:
            continue
        sub = df[[key, tcol] + [c for c in cols if DEMO_RE.match(cols[c]) and c != key and c != tcol]].copy()
        sub.columns = ['_name', 'target'] + [f'demo_{i}' for i in range(len(sub.columns) - 2)]
        sub['_patient'] = sub._name.astype(str).str.strip().str.casefold().map(names)
        sub = sub[sub._patient.notna()]
        if len(sub) < 50:
            continue
        sub['target'] = pd.to_numeric(sub.target, errors='coerce')
        sub = sub[sub.target.notna()]
        if len(sub) < 50:
            continue
        sub['sheet'] = sheet
        frames.append(sub)
        used.append((sheet, tcol, len(sub)))
    return (pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()), used


def grouped_ridge(X, y, groups, folds=5, alphas=(1e1, 1e2, 1e3, 1e4, 1e5)):
    from sklearn.linear_model import Ridge
    from sklearn.impute import SimpleImputer
    from sklearn.model_selection import GroupKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    def estimator(alpha):
        return make_pipeline(SimpleImputer(strategy='median', keep_empty_features=True),
                             StandardScaler(), Ridge(alpha=alpha))

    X, y, groups = np.asarray(X), np.asarray(y), np.asarray(groups)
    if len(X) != len(y) or len(groups) != len(y):
        raise ValueError('Features, targets and groups must align')
    pred, chosen = np.full(len(y), np.nan), []
    for tr, te in GroupKFold(n_splits=folds).split(X, y, groups):
        inner = GroupKFold(min(4, len(np.unique(groups[tr]))))
        candidates = []
        for a in alphas:
            ip = np.full(len(tr), np.nan)
            for it, iv in inner.split(X[tr], y[tr], groups[tr]):
                m = estimator(a)
                m.fit(X[tr[it]], y[tr[it]])
                ip[iv] = m.predict(X[tr[iv]])
            score = spearman(ip, y[tr])
            candidates.append((score if np.isfinite(score) else -np.inf, a))
        a = max(candidates)[1]
        m = estimator(a)
        m.fit(X[tr], y[tr])
        pred[te] = m.predict(X[te])
        chosen.append(a)
    return spearman(pred, y), chosen, pred


def aligned_features(values, joined):
    """Keep original feature row identity across folder filtering and joins."""
    rows = joined['_feature_row'].to_numpy(dtype=int)
    if np.any(rows < 0) or np.any(rows >= len(values)):
        raise ValueError('Feature row outside source matrix')
    return np.asarray(values)[rows].astype(np.float64)


def spearman(a, b):
    from scipy.stats import spearmanr
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 10:
        return float('nan')
    return float(spearmanr(a[ok], b[ok]).statistic)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workbook', type=Path, required=True)
    parser.add_argument('--feature-dir', type=Path, required=True)
    parser.add_argument('--models', nargs='+', required=True)
    parser.add_argument('--folder', default='LumbarP')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('Choose a new output directory')
    args.output.mkdir(parents=True)

    sheets = load_workbook(args.workbook)
    print('sheets:', {k: list(map(str, v.columns))[:12] for k, v in sheets.items()})

    results = {}
    for model in args.models:
        f = args.feature_dir / f'features_{model}.npy'
        if not f.exists():
            print(f'skip {model}: no features'); continue
        meta = pd.read_csv(args.feature_dir / f'meta_{model}.csv')
        meta['_feature_row'] = np.arange(len(meta))
        meta['person_group'] = meta.patient.map(person_group)
        meta = meta[meta.folder == args.folder].reset_index(drop=True)
        values = np.load(f)
        tbl, used = find_targets(sheets, meta)
        if tbl.empty:
            print(f'{model}: no usable target'); results[model] = {'status': 'no_target'}; continue
        tbl = tbl.drop_duplicates('patient' if 'patient' in tbl.columns else '_patient')
        key = '_patient'
        tbl = tbl.groupby(key).first().reset_index()
        j = meta.merge(tbl, left_on='patient', right_on=key, how='inner')
        X = aligned_features(values, j)
        print(f'{model}: {len(j)} images / {j.patient.nunique()} patients joined, target sheets {used}')
        if j.person_group.nunique() < 40:
            results[model] = {'status': 'too_few_patients', 'patients': int(j.patient.nunique())}; continue

        demo_cols = [c for c in j.columns if str(c).startswith('demo_')]
        y = j.target.to_numpy(float)
        r_feat, alpha, pred = grouped_ridge(X, y, j.person_group.to_numpy())
        r_demo = None
        if demo_cols:
            D = j[demo_cols].apply(pd.to_numeric, errors='coerce')
            D = pd.get_dummies(D, drop_first=True).astype(float)
            keep = D.notna().any(axis=1).to_numpy()
            if keep.sum() > 40:
                r_demo, _, _ = grouped_ridge(D.to_numpy(float)[keep], y[keep], j.person_group.to_numpy()[keep])
        results[model] = {'status': 'ok', 'images': int(len(j)), 'patients': int(j.patient.nunique()),
                          'target_sheets': used, 'ridge_alpha_by_outer_fold': list(map(float, alpha)),
                          'split_groups': int(j.person_group.nunique()),
                          'patient_identity_verified': False,
                          'spearman_features': r_feat,
                          'spearman_demographics_only': r_demo,
                          'spearman_gain': None if r_demo is None or not np.isfinite(r_demo) else r_feat - r_demo,
                          'n_demo_columns': len(demo_cols)}
        j.assign(pred=pred, target=y).to_csv(args.output / f'probe_{model}.csv', index=False)

    dump(args.output / 'report.json', {'protocol': __doc__, 'folder': args.folder, 'results': results,
                                       'clinical_validation': False, 'model_affects_decision': False})
    print(json.dumps(results, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
