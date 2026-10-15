"""Bind source-folder evaluation groups to current DICOM study/pixel identities.

Folder group keys can differ from de-identified StudyInstanceUIDs. No QC target
values are inspected. Cross-group or cross-fold identity collisions are fatal.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import warnings

import joblib

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from dxaqc.dicom_io import read_dxa
from gpu_research.common import digest


def bind_identity(mapping, key, group, fold):
    identity = (str(group), int(fold))
    if key in mapping and mapping[key] != identity:
        raise ValueError('Source image/study identity crosses evaluation groups or folds')
    mapping[key] = identity


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('bank', 'dataset', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('Choose a new replay manifest')
    manifest = json.loads((args.bank / 'manifest.json').read_text())
    if manifest.get('full_data_fit'):
        raise ValueError('A full-data model has no held-out replay identities')
    if digest(args.bank / 'bank.joblib') != manifest['bank_sha256']:
        raise ValueError('Head bank checksum differs')
    value = joblib.load(args.bank / 'bank.joblib')
    by_study, by_pixel, group_uids = {}, {}, {}
    warnings.filterwarnings('ignore', message='Invalid value for VR UI:.*')
    for row in value['baseline'].itertuples():
        path = (args.dataset / row.source_path).resolve()
        if not path.is_relative_to(args.dataset.resolve()):
            raise ValueError('Evaluation source path escapes dataset')
        image = read_dxa(path)
        bind_identity(by_study, image.study_uid, row.study, row.fold)
        bind_identity(by_pixel, image.pixel_sha256, row.study, row.fold)
        group_uids.setdefault(str(row.study), set()).add(image.study_uid)
    report = {'schema_version': 1, 'bank_sha256': manifest['bank_sha256'],
              'source_fingerprint': manifest['source_fingerprint'],
              'dicom_study_to_fold': {key: identity[1] for key, identity in by_study.items()},
              'decoded_pixel_to_fold': {key: identity[1] for key, identity in by_pixel.items()},
              'source_groups': len(group_uids), 'dicom_studies': len(by_study), 'decoded_images': len(by_pixel),
              'source_groups_with_multiple_dicom_studies': sum(len(v) > 1 for v in group_uids.values()),
              'cross_group_identity_collisions': 0, 'clinical_validation': False,
              'qc_target_values_used': False}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({k: v for k, v in report.items() if not isinstance(v, dict)}))


if __name__ == '__main__':
    main()
