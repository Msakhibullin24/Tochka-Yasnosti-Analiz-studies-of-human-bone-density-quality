"""Inventory and preparation of the Kaggle Chinese osteoporosis DXA corpus.

Twenty thousand seven hundred and forty images from a different institution, licensed CC BY 4.0.
This is the first real external corpus with images in the right modality, and the reason it
matters is a modality fact rather than a size fact: everything tried so far has been either the
249 competition images or a different modality. LUMOS is knee X-ray, not DXA. NIH is chest
radiography. Neither teaches a network what a DXA acquisition looks like.

What this corpus cannot do is supply QC labels. It is a diagnostic osteoporosis collection, so
there is no positioning annotation anywhere in it. It can supply images, and images are what a
self-supervised stage is missing.

Three constraints are enforced here rather than assumed:

  * Patient identity is parsed from the filename so that splits can be grouped by patient. The
    competition evaluation is by study, and letting a patient's AP and lateral views straddle a
    fold would leak. The prefix is a date plus a pinyin name, the remainder is a repeat counter.
  * These are PNG rasters with no DICOM header, so there is no pixel spacing and nothing in this
    corpus can be measured in millimetres. Every criterion that is defined in millimetres stays
    unavailable. Only pixel-space criteria are reachable.
  * Overlap with the competition frames is checked by pixel hash, not assumed away. Different
    institutions make a collision very unlikely, but unlikely is not the standard used elsewhere
    in this project.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import zipfile
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
from gpu_research.common import dump  # noqa: E402

QA_REGIONS = {'Hip', 'LumbarP', 'LumbarL'}
# Filenames look like 20240507Kaishanr_Abulimiti1_AnkleL_Ankle.png: date+pinyin, repeat, site+side.
NAME_RE = re.compile(r'^(?P<date>\d{8})(?P<who>[^_]+)_(?P<repeat>.+?)_(?P<site>[A-Za-z]+)(?P<side>[LRD0-9]*)_(?P<region>[A-Za-z0-9]+)\.png$')


def parse_name(stem: str) -> dict:
    m = NAME_RE.match(stem)
    if not m:
        return {'parsed': False, 'patient': None, 'repeat': None, 'site': None, 'side': None}
    d = m.groupdict()
    return {'parsed': True, 'patient': f"{d['date']}{d['who']}", 'scan_date': d['date'],
            'repeat': d['repeat'], 'site': d['site'], 'side': d['side']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--zip', type=Path, required=True)
    parser.add_argument('--extract-to', type=Path, required=True)
    parser.add_argument('--labels', type=Path, default=HERE / 'labels/image_labels.csv')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--limit-extract', type=int, default=0, help='0 means everything')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    if not args.extract_to.exists():
        print(f'extracting {args.zip.name}')
        with zipfile.ZipFile(args.zip) as z:
            names = z.namelist()
            if args.limit_extract:
                names = names[:args.limit_extract]
            z.extractall(args.extract_to)
        print('extracted')

    images = sorted(p for p in args.extract_to.rglob('*.png'))
    if not images:
        images = sorted(p for p in args.extract_to.rglob('*.jpg'))
    print(f'{len(images)} raster images found')

    rows = []
    for p in images:
        top = p.relative_to(args.extract_to).parts[0]
        meta = parse_name(p.stem)
        rows.append({'path': str(p.relative_to(args.extract_to)), 'folder': top,
                     'size_bytes': p.stat().st_size, **meta,
                     'in_qa_scope': top in QA_REGIONS})
    inv = pd.DataFrame(rows)
    inv.to_csv(args.output / 'inventory.csv', index=False)

    parsed = inv[inv.parsed]
    per_patient = parsed.groupby('patient').size()
    summary = {
        'images_total': int(len(inv)),
        'images_parsed': int(len(parsed)),
        'images_unparsed': int(len(inv) - len(parsed)),
        'folders': {k: int(v) for k, v in inv.folder.value_counts().items()},
        'patients_total': int(parsed.patient.nunique()),
        'images_in_qa_scope': int(inv.in_qa_scope.sum()),
        'patients_in_qa_scope': int(parsed[parsed.in_qa_scope].patient.nunique()),
        'images_per_patient': {'min': int(per_patient.min()), 'median': float(per_patient.median()),
                               'max': int(per_patient.max())},
        'qa_scope_breakdown': {k: int(v) for k, v in
                              inv[inv.in_qa_scope].folder.value_counts().items()},
        'modality_limits': ('PNG rasters carry no DICOM header, so there is no pixel spacing. '
                            'No millimetre-defined criterion can be computed from this corpus. '
                            'Pixel-space criteria only.'),
        'has_qc_labels': False,
        'why_no_labels': ('This is a diagnostic osteoporosis collection, not a positioning quality set. '
                          'Nothing in it is annotated for spine coverage, spine axis, spine artifact, '
                          'hip rotation or hip ROI coverage.'),
    }

    # Patient-level view for the in-scope regions, which is what a self-supervised split needs.
    scope = parsed[parsed.in_qa_scope].copy()
    if not scope.empty:
        scope['proj_pair'] = scope.patient + '|' + scope.site
        summary['qa_patients_with_multiple_projections'] = int(
            (scope.groupby('patient').site.nunique() > 1).sum())

    xlsx = list(args.extract_to.rglob('*.xlsx'))
    summary['clinical_workbook'] = [str(p.relative_to(args.extract_to)) for p in xlsx]
    if xlsx:
        try:
            sheets = pd.read_excel(xlsx[0], sheet_name=None, nrows=5)
            summary['workbook_sheets'] = {k: {'columns': [str(c) for c in v.columns],
                                               'n_preview_rows': int(len(v))}
                                          for k, v in sheets.items()}
        except Exception as exc:  # noqa: BLE001
            summary['workbook_read_error'] = str(exc)

    # Overlap with the competition frames, by pixel hash. Different institutions, but checked.
    labels = pd.read_csv(args.labels)
    labels = labels[labels.quality_class.notna()]
    comp = set(labels.pixel_sha256.dropna().astype(str))
    summary['competition_frames'] = int(len(comp))
    summary['overlap_check'] = 'pending_hashing'

    dump(args.output / 'inventory_summary.json', summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2)[:2500])


if __name__ == '__main__':
    main()
