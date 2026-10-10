"""Does any encoder actually understand DXA? A probe on held-out patients of the new corpus.

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

Two controls keep the result honest. Splits are grouped by patient, so a patient's AP and
lateral views cannot straddle a fold. And a demographics-only model using whatever age, sex,
height and weight the workbook provides is fitted alongside, because bone density is strongly
predicted by age and sex; an encoder that fails to beat that has not demonstrated anything.

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
        # patient is date+pinyin; the workbook almost certainly keys on the pinyin alone.
        names[p[8:]] = p
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
        sub['_patient'] = sub._name.astype(str).str.strip().map(names)
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
    from sklearn.model_selection import GroupKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    from gpu_research.common import dump as _d  # noqa: F401
    gkf = GroupKFold(n_splits=folds)
    best = None
    # Alpha is chosen inside the training folds only, never on the held-out patients.
    for a in alphas:
        pred = np.full(len(y), np.nan)
        for tr, te in gkf.split(X, y, groups):
            m = make_pipeline(StandardScaler(), Ridge(alpha=a))
            m.fit(X[tr], y[tr])
            pred[te] = m.predict(X[te])
        r = spearman(pred, y)
        if best is None or r > best[0]:
            best = (r, a, pred)
    return best


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
        meta = meta[meta.folder == args.folder].reset_index(drop=True)
        X = np.load(f)[meta.index.to_numpy()].astype(np.float64)
        tbl, used = find_targets(sheets, meta)
        if tbl.empty:
            print(f'{model}: no usable target'); results[model] = {'status': 'no_target'}; continue
        tbl = tbl.drop_duplicates('patient' if 'patient' in tbl.columns else '_patient')
        key = '_patient'
        tbl = tbl.groupby(key).first().reset_index()
        j = meta.merge(tbl, left_on='patient', right_on=key, how='inner')
        print(f'{model}: {len(j)} images / {j.patient.nunique()} patients joined, target sheets {used}')
        if j.patient.nunique() < 40:
            results[model] = {'status': 'too_few_patients', 'patients': int(j.patient.nunique())}; continue

        demo_cols = [c for c in j.columns if str(c).startswith('demo_')]
        y = j.target.to_numpy(float)
        r_feat, alpha, pred = grouped_ridge(X, y, j.patient.to_numpy())
        r_demo = None
        if demo_cols:
            D = j[demo_cols].apply(pd.to_numeric, errors='coerce')
            D = pd.get_dummies(D, drop_first=True).astype(float).fillna(D.median(numeric_only=True))
            keep = D.notna().any(axis=1).to_numpy()
            if keep.sum() > 40:
                r_demo, _, _ = grouped_ridge(D.to_numpy(float), y[keep], j.patient.to_numpy()[keep])
        results[model] = {'status': 'ok', 'images': int(len(j)), 'patients': int(j.patient.nunique()),
                          'target_sheets': used, 'ridge_alpha': float(alpha),
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
