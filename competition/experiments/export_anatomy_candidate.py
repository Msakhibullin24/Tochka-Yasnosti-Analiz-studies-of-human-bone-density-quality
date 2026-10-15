"""Export fitted candidate heads without training tables, targets or image features.

Copies encoder provenance inline so inference has no dependency on training
feature files. Exported hashes refer to the reduced artifacts; original hashes
remain explicit provenance. No weights are promoted to the released profile.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import joblib

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from gpu_research.common import digest


HEAD_FIELDS = ('fold', 'group', 'heads', 'columns', 'codes')


def export_bank(source, target, *, feature_hash=None):
    manifest = json.loads((source / 'manifest.json').read_text())
    if digest(source / 'bank.joblib') != manifest['bank_sha256']:
        raise ValueError('Source model bank checksum differs')
    source_value = joblib.load(source / 'bank.joblib')
    value = {'schema_version': 2, 'inference_only': True,
             'folds': [{key: state[key] for key in HEAD_FIELDS} for state in source_value['folds']]}
    target.mkdir(parents=True)
    joblib.dump(value, target / 'bank.joblib', compress=3)
    manifest['training_bank_sha256'] = manifest['bank_sha256']
    manifest['bank_sha256'] = digest(target / 'bank.joblib')
    manifest['inference_only'] = True
    if feature_hash is not None:
        manifest['training_extra_features_sha256'] = manifest['extra_features_sha256']
        manifest['extra_features_sha256'] = feature_hash
    for entry in manifest['embeddings']:
        entry['metadata'] = json.loads(Path(entry.pop('path')).with_suffix('.json').read_text())
        # Configuration/checkpoint identity and precision suffice for inference.
        entry['metadata'] = {key: entry['metadata'][key] for key in
                             ('weights', 'config_sha256', 'processor_sha256', 'precision', 'size')}
    (target / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('bank', 'ranking_bank', 'decoders', 'feature_spec', 'output'):
        parser.add_argument('--' + name.replace('_', '-'), type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('Choose a new export directory')
    args.output.mkdir(parents=True)
    features = json.loads(args.feature_spec.read_text())
    original_feature_sha = digest(args.feature_spec)
    if original_feature_sha != json.loads((args.bank / 'manifest.json').read_text())['extra_features_sha256']:
        raise ValueError('Feature spec differs from training bank')
    features.pop('entries', None)
    (args.output / 'feature_spec.json').write_text(json.dumps(features, indent=2) + '\n')
    current = export_bank(args.bank, args.output / 'bank', feature_hash=digest(args.output / 'feature_spec.json'))
    reference = export_bank(args.ranking_bank, args.output / 'ranking_bank')
    evaluation = json.loads((args.decoders / 'evaluation.json').read_text())
    if (evaluation['bank_sha256'] != current['training_bank_sha256'] or
            evaluation['ranking_bank_sha256'] != reference['training_bank_sha256']):
        raise ValueError('Decoder source identities differ')
    # Only fitted threshold/decoder state is exported, no OOF arrays or QC targets.
    specs = joblib.load(args.decoders / 'decoders.joblib')
    decoders = [{key: state[key] for key in (('fold', 'group', 'criterion_decoder', 'quality_decoder', 'quality_cut', 'preserve_independent')
                if 'criterion_decoder' in state else ('fold', 'group', 'decoder', 'ranking_blend'))} for state in specs]
    (args.output / 'decoders').mkdir()
    joblib.dump(decoders, args.output / 'decoders/decoders.joblib', compress=3)
    protocol = {'schema_version': 2, 'inference_only': True, 'bank_sha256': current['bank_sha256'],
                'ranking_bank_sha256': reference['bank_sha256'],
                'training_evaluation_sha256': digest(args.decoders / 'evaluation.json'),
                'decoder_sha256': digest(args.output / 'decoders/decoders.joblib'),
                'clinical_validation': False, 'automatic_promotions': 0}
    (args.output / 'decoders/evaluation.json').write_text(json.dumps(protocol, indent=2) + '\n')
    inventory = {str(p.relative_to(args.output)): digest(p) for p in args.output.rglob('*') if p.is_file()}
    (args.output / 'export.json').write_text(json.dumps({'schema_version': 1, 'files': inventory,
        'training_tables_exported': False, 'target_labels_exported': False, 'cached_image_features_exported': False,
        'original_feature_spec_sha256': original_feature_sha, 'code_sha256': digest(Path(__file__))}, indent=2) + '\n')
    print(json.dumps({'files': len(inventory), 'output': str(args.output)}), flush=True)


if __name__ == '__main__':
    main()
