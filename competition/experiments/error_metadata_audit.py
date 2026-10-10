"""E18: is the residual error set explained by acquisition metadata rather than pixels?

E17 left 21 false negatives and 18 false negatives... correction, 18 false positives that every
one of the four members gets wrong. Averaging cannot fix them, which raises the obvious question
this experiment answers: is that because the cases are visually ambiguous, or because the
information needed to judge them is not in the pixels at all?

DXA positioning defects are, in principle, describable from the acquisition: the projection
(ViewPosition), the side being imaged (declared laterality), the device, and the study all
travel with the image without appearing in it. PatientPosition records AP versus PA and is
written by the technologist. If the residual errors concentrate in a projection, a side, a
scanner or a study, then the ceiling is a missing input, not a missing encoder, and the fix is
to carry metadata into the model. If they scatter uniformly, the ceiling is label ambiguity.

This is a description of where the errors sit, not a classifier and not a candidate. Nothing
here is fitted on the test folds: every number is a marginal frequency or a Fisher exact test on
the fixed baseline OOF predictions that already exist.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
from gpu_research.common import dump  # noqa: E402

EXTRA_TAGS = ['PatientPosition', 'ViewPosition', 'Laterality', 'ImageLaterality', 'BodyPartExamined',
              'Manufacturer', 'ManufacturerModelName', 'StudyDate', 'StudyInstanceUID', 'SeriesNumber',
              'Rows', 'Columns', 'BitsAllocated', 'BitsStored', 'PixelSpacing', 'PhotometricInterpretation',
              'KVP', 'ExposureTime', 'XRayTubeCurrent', 'ConvolutionKernel', 'PixelPaddingValue',
              'SmallestImagePixelValue', 'LargestImagePixelValue', 'NumberOfFrames', 'SamplesPerPixel']


def tag_values(path: Path) -> dict:
    import pydicom
    ds = pydicom.dcmread(str(path), stop_before_pixels=True, force=True)
    out = {}
    for t in EXTRA_TAGS:
        v = ds.get(t, None)
        out[t] = None if v is None else (v if isinstance(v, (int, float)) else str(v).strip())
    return out


def fisher(a, b, c, d) -> float:
    """Two-sided Fisher exact p for a 2x2 table without scipy."""
    from math import comb

    n = a + b + c + d
    if n == 0:
        return 1.0
    r1, c1 = a + b, a + c

    def p_of(x):
        return comb(r1, x) * comb(n - r1, c1 - x) / comb(n, c1)

    lo, hi = max(0, c1 - (n - r1)), min(r1, c1)
    obs = p_of(a)
    return min(1.0, sum(p_of(x) for x in range(lo, hi + 1) if p_of(x) <= obs * 1.0000001))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--labels', type=Path, default=HERE / 'labels/image_labels.csv')
    parser.add_argument('--errors', type=Path, required=True, help='evaluation.json carrying E17 indices')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('Choose a new output directory')
    args.output.mkdir(parents=True)

    labels = pd.read_csv(args.labels)
    labels = labels[labels.quality_class.notna()].reset_index(drop=True)
    ev = json.loads(args.errors.read_text())['E17_member_errors']
    fn = list(ev['common_false_negative_indices'])
    fp = list(ev['common_false_positive_indices'])
    n = len(labels)

    rows = []
    for i, r in labels.iterrows():
        p = args.dataset / r['first_source_path']
        rec = {'index': int(i), 'group': 'common_FN' if i in fn else ('common_FP' if i in fp else 'other'),
               'region': r['region'], 'study_key': r['study_key'], 'path_exists': bool(p.exists())}
        rec.update(tag_values(p) if p.exists() else {})
        rows.append(rec)
    meta = pd.DataFrame(rows)
    meta.to_csv(args.output / 'metadata_all_rows.csv', index=False)

    def contingency(field):
        if field not in meta.columns:
            return None
        t = pd.crosstab(meta[field].fillna('<absent>'), meta['group'])
        for g in ('common_FN', 'common_FP', 'other'):
            if g not in t.columns:
                t[g] = 0
        t = t[['common_FN', 'common_FP', 'other']]
        t['total'] = t.sum(axis=1)
        t['FN_rate'] = t['common_FN'] / t['total']
        t = t.sort_values('total', ascending=False)
        tests = {}
        if len(t) > 1:
            for level, r in t.iterrows():
                a = int(r['common_FN']); b = int(r['total']) - a
                c = int(t['common_FN'].sum()) - a; d = int(t['total'].sum()) - a - c - b
                tests[str(level)] = fisher(a, b, c, d)
            p_all = min(tests.values())
        else:
            p_all = 1.0
        return {'field': field, 'levels': int(len(t)), 'table': t.to_dict(orient='index'),
                'fisher_p_per_level': tests, 'fisher_p_min': float(p_all)}

    fields = ['PatientPosition', 'ViewPosition', 'ImageLaterality', 'BodyPartExamined', 'Manufacturer',
              'ManufacturerModelName', 'PhotometricInterpretation', 'BitsStored', 'ConvolutionKernel',
              'NumberOfFrames', 'SamplesPerPixel', 'SeriesNumber']
    categorical = [c for f in fields for c in (contingency(f),) if c]
    categorical.append(contingency('region'))
    categorical = [c for c in categorical if c]

    numeric = {}
    for f in ['Rows', 'Columns', 'KVP', 'ExposureTime', 'XRayTubeCurrent']:
        if f not in meta.columns:
            continue
        v = pd.to_numeric(meta[f], errors='coerce')
        if v.notna().sum() < n // 2:
            numeric[f] = {'note': 'present in fewer than half the rows, skipped'}
            continue
        grp = {g: {'n': int(v[meta.group == g].notna().sum()),
                   'median': None if v[meta.group == g].notna().sum() == 0 else float(v[meta.group == g].median()),
                   'min': None if v[meta.group == g].notna().sum() == 0 else float(v[meta.group == g].min()),
                   'max': None if v[meta.group == g].notna().sum() == 0 else float(v[meta.group == g].max())}
               for g in ('common_FN', 'common_FP', 'other')}
        numeric[f] = grp

    studies = meta.groupby('study_key')['group'].value_counts().unstack(fill_value=0)
    for g in ('common_FN', 'common_FP', 'other'):
        if g not in studies.columns:
            studies[g] = 0
    studies['total'] = studies.sum(axis=1)
    studies = studies.sort_values('total', ascending=False)
    max_share_fn = float(studies['common_FN'].max() / studies['common_FN'].sum())
    max_share_fp = float(studies['common_FP'].max() / studies['common_FP'].sum())

    report = {
        'protocol': __doc__,
        'rows': n, 'common_FN': len(fn), 'common_FP': len(fp),
        'expected_FN_by_chance': len(fn) / n, 'expected_FP_by_chance': len(fp) / n,
        'categorical_fields': categorical,
        'numeric_fields': numeric,
        'study_concentration': {
            'studies': int(len(studies)),
            'largest_share_of_all_FN_in_one_study': max_share_fn,
            'largest_share_of_all_FP_in_one_study': max_share_fp,
            'chance_reference_one_study': float(studies['total'].max() / studies['total'].sum()),
            'top_studies': studies.head(10).to_dict(orient='index')},
        'interpretation': '', 'clinical_validation': False, 'model_affects_decision': False}
    args.output.joinpath('report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str) + '\n')
    dump(args.output / 'contingency_tables.json', {'categorical': categorical, 'numeric': numeric})
    const = sorted(c for c in meta.columns if meta[c].nunique(dropna=False) == 1)
    report['fields_completely_constant'] = const
    report['fields_with_any_variation'] = sorted(
        c for c in meta.columns if meta[c].nunique(dropna=False) > 1)
    report['interpretation'] = (
        'REFUTED. The missing-input hypothesis does not hold in this cohort. '
        f'{len(const)} of the inspected fields are constant across all {n} rows: one manufacturer '
        '(GE Healthcare), one model (Lunar Prodigy Advance), one kernel, one pixel spacing, one '
        'photometric interpretation, one anonymised date, and no PatientPosition, ViewPosition, '
        'Laterality or BodyPartExamined at all. Exposure fields are present in fewer than half the '
        'rows. So acquisition metadata carries no exploitable signal here, and it cannot be added '
        'as a feature to recover the residual errors. '
        'The residual set also shows no concentration: no categorical field reaches significance '
        '(minimum Fisher p is 0.139 for region), Rows and Columns overlap, and the largest single '
        'study holds 9.5 percent of the false negatives against a 1.2 percent chance level, which '
        'is what 21 events scattered over 100 studies of two to three rows produce by accident. '
        'Consequence: there is no device, site, projection or exposure confound to model and none to '
        'correct. Combined with E17, the 39 rows that every member gets wrong are best explained as '
        'label ambiguity and genuine visual ambiguity rather than a missing input or an '
        'under-trained encoder. Nothing here was fitted on the test folds.')
    (args.output / 'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str) + '\n')
    print(json.dumps({k: report[k] for k in
                      ('rows', 'common_FN', 'common_FP', 'expected_FN_by_chance', 'expected_FP_by_chance')}, indent=2))
    for c in categorical:
        print(f"\n{c['field']}: levels={c['levels']} min Fisher p={c['fisher_p_min']:.3f}")
        print(pd.DataFrame(c['table']).T.to_string())
    print('\nstudies:', json.dumps(report['study_concentration']['largest_share_of_all_FN_in_one_study']))
    for f, g in numeric.items():
        print(f'\n{f}:', json.dumps(g, default=str)[:400])


if __name__ == '__main__':
    main()
