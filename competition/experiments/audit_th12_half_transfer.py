"""Verify source identity and score the research half-height candidate on original DXA."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
import warnings

import joblib
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dxaqc.dicom_io import read_dxa
from experiments.train_th12_half_classifier import assess


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(args):
    if args.output.exists():
        raise ValueError('Choose a new output report')
    torch.set_num_threads(4)
    bundle = joblib.load(args.model)
    source = json.loads(args.source_manifest.read_text())
    if source.get('review_package_version') != 4:
        raise ValueError('Expected source-identity review package v4')
    root = args.dataset.resolve()
    cases, seen = [], set()
    for reference in source['cases']:
        if 'Th12' not in reference.get('landmarks', {}):
            continue
        path = (root/reference['path_to_file']).resolve()
        if not path.is_relative_to(root) or sha(path) != reference['source_sha256']:
            raise ValueError('DICOM source hash mismatch')
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            image = read_dxa(path)
        if (image.image_uid != reference['image_uid'] or image.pixel_sha256 != reference['pixel_sha256'] or
                image.pixels.shape != (reference['height'], reference['width'])):
            raise ValueError('DICOM pixel identity or geometry mismatch')
        if image.pixel_sha256 in seen:
            raise ValueError('Duplicate unique-source pixel identity')
        seen.add(image.pixel_sha256)
        result = assess(bundle, image.pixels)
        cases.append({'image_uid': image.image_uid, 'pixel_sha256': image.pixel_sha256,
                      'source_sha256': reference['source_sha256'], 'path_to_file': reference['path_to_file'],
                      **result, 'reference': None})
        if len(cases) % 25 == 0:
            print(f'Checked {len(cases)} DXA spine images', flush=True)
    if len(cases) != 99:
        raise ValueError('Unexpected unique DXA spine cohort')
    report = {'images': len(cases), 'prediction_counts': dict(Counter(c['candidate_rule'] for c in cases)),
              'accuracy': None, 'clinical_validation': False, 'release_modified': False,
              'requirements_complete': False, 'cases': cases,
              'source_manifest_sha256': sha(args.source_manifest), 'model_sha256': sha(args.model),
              'code_sha256': sha(Path(__file__)),
              'limitations': ['No independently reviewed full Th12 body geometry on organizer DXA.',
                              'Source-only conformal threshold has no demonstrated DXA coverage.',
                              'Predicted classes are exploratory and never clinical verdicts.']}
    args.output.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({k: report[k] for k in ('images','prediction_counts','accuracy')}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('model','source-manifest','dataset','output'):
        parser.add_argument('--'+name, type=Path, required=True)
    run(parser.parse_args())
