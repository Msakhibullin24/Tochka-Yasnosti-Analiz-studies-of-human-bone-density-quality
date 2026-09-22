"""Inventory downloaded NHANES XPT tables without treating rows as scan images."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--assets', type=Path, default=Path('data/specialists'))
    args = parser.parse_args()
    catalog = json.loads((Path(__file__).resolve().parents[1]/'docs/competition/specialist_sources.json').read_text())
    tables = []
    for artifact in catalog['artifacts']:
        if artifact['kind'] != 'tabular_data':
            continue
        path = args.assets/artifact['path']
        if hashlib.sha256(path.read_bytes()).hexdigest() != artifact['sha256']:
            raise ValueError(f"Checksum mismatch: {artifact['id']}")
        frame = pd.read_sas(path, format='xport')
        if 'SEQN' not in frame:
            raise ValueError('NHANES table lacks participant key')
        tables.append({'id': artifact['id'], 'cycle': artifact['cycle'], 'records': len(frame),
                       'participants': int(frame.SEQN.nunique()), 'columns': list(frame.columns),
                       'imputation_columns': [c for c in frame if 'mult' in c.lower()],
                       'image_files': 0})
    report = {'tables': tables, 'image_training_ready': False,
              'note': 'Participant counts overlap across tables. Repeated imputation records are not scans.'}
    (args.assets/'nhanes-inventory.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps([{k: v for k,v in row.items() if k != 'columns'} for row in tables], indent=2))


if __name__ == '__main__':
    main()
