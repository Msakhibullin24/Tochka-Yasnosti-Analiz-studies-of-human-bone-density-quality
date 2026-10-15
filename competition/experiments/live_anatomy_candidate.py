"""Raw-DICOM inference for the measured anatomical QC research candidate.

Uses saved heads/decoders only: no fitting, labels or cached image features at
inference. Study-held-out replay is explicit; default new-study inference uses
the five fitted folds as an ensemble. The latter has no independent estimate
on studies used for training. This does not replace the released profile.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import socket
import warnings
import hashlib

import joblib
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from dxaqc.decision import decide
from dxaqc.learned_anatomy import mask_regions, roi_proposals
from dxaqc.model import _matrix, group_of
from dxaqc.pipeline import Analyzer, Options, run_batch
from experiments.mask_anatomy_features import spine_features
from experiments.three_iteration_upgrade import predict_family, decode_scores
from gpu_research.common import digest, image_input, load_encoder, representations

EXECUTION_SOURCE = Path(__file__).read_bytes()
EXECUTION_SHA256 = hashlib.sha256(EXECUTION_SOURCE).hexdigest()


def anatomical_axis_features(features, angle, enabled):
    updated = dict(features)
    if enabled:
        updated['spine_abs_angle_deg'] = abs(float(angle)) if angle is not None else None
    return updated


def load_bank(directory):
    manifest = json.loads((directory / 'manifest.json').read_text())
    if digest(directory / 'bank.joblib') != manifest['bank_sha256']:
        raise ValueError('Fitted head-bank checksum differs')
    value = joblib.load(directory / 'bank.joblib')
    folds = {(s['fold'], s['group']): {k: s[k] for k in ('fold', 'group', 'heads', 'columns', 'codes')}
             for s in value['folds']}
    # Keep only study->fold identities for evaluation. No QC labels, inner OOF
    # scores or reference image features are retained in the inference object.
    studies = dict(zip(value['baseline'].study.astype(str), value['baseline'].fold.astype(int)))
    return manifest, folds, studies


class LiveAnatomyCandidate:
    def __init__(self, bank, ranking_bank, decoders, feature_spec, anatomy_model,
                 encoder_directories, *, held_out_replay=False, replay_manifest=None):
        os.environ['HF_HUB_OFFLINE'] = '1'
        os.environ['TRANSFORMERS_OFFLINE'] = '1'
        self.base = Analyzer()
        self.bundle = self.base.bundle
        self.held_out_replay = held_out_replay
        self.manifest, self.folds, self.source_groups = load_bank(bank)
        self.replay_manifest = None
        if held_out_replay:
            if replay_manifest is None:
                raise ValueError('Held-out replay requires an audited DICOM identity manifest')
            replay = json.loads(replay_manifest.read_text())
            if (replay.get('schema_version') != 1 or replay.get('bank_sha256') != self.manifest['bank_sha256'] or
                    replay.get('source_fingerprint') != self.manifest['source_fingerprint'] or
                    replay.get('cross_group_identity_collisions') != 0):
                raise ValueError('Replay source identity differs')
            self.replay_manifest = replay
        old_manifest, self.reference, _ = load_bank(ranking_bank)
        if any(self.manifest[k] != old_manifest[k] for k in ('labels_sha256', 'baseline_sha256', 'source_fingerprint')):
            raise ValueError('Quality-reference identities differ')
        evaluation = json.loads((decoders / 'evaluation.json').read_text())
        if evaluation['bank_sha256'] != self.manifest['bank_sha256'] or evaluation['ranking_bank_sha256'] != old_manifest['bank_sha256']:
            raise ValueError('Decoder head-bank identity differs')
        self.decoders = {(s['fold'], s['group']): s for s in joblib.load(decoders / 'decoders.joblib')}
        if any('decoder' not in s or s['decoder']['calibrator'] is not None for s in self.decoders.values()):
            raise ValueError('This raw prototype supports the typed-union candidate with an explicit ranking blend')
        self.workflow_profile = {'profile_id': 'research-live-anatomy-qc-20261015', 'clinical_validation': False,
                                 'status': 'research_only', 'source_profile_id':
                                 (self.base.workflow_profile or {}).get('profile_id'),
                                 'evaluation_mode': 'study_held_out_replay' if held_out_replay else 'fold_ensemble'}
        self.records = []
        extra = json.loads(feature_spec.read_text())
        if digest(feature_spec) != self.manifest['extra_features_sha256'] or extra['columns_by_group']['hip']:
            raise ValueError('This measured prototype uses the spine-anatomy feature bank')
        if digest(anatomy_model / 'fold_0/checkpoint_best.pth') != extra['checkpoint_sha256']:
            raise ValueError('Anatomy checkpoint differs from feature generation')
        from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
        self.anatomy = nnUNetPredictor(use_mirroring=False, allow_tqdm=False, verbose=False,
                                      device=torch.device('cuda'))
        self.anatomy.initialize_from_trained_model_folder(str(anatomy_model), use_folds=(0,),
                                                         checkpoint_name='checkpoint_best.pth')
        self.anatomy_sha256 = extra['checkpoint_sha256']
        self.encoders = []
        for directory, embedding in zip(encoder_directories, self.manifest['embeddings'], strict=True):
            metadata = json.loads(Path(embedding['path']).with_suffix('.json').read_text())
            for name, checksum in metadata['weights'].items():
                if digest(directory / name) != checksum:
                    raise ValueError('Frozen encoder checkpoint differs')
            kind = json.loads((directory / 'config.json').read_text())['model_type']
            processor, encoder = load_encoder(directory, kind, 'cuda', metadata['precision'])
            self.encoders.append((processor, encoder, kind, metadata['size']))
        torch.set_num_threads(4)

    @staticmethod
    def scores(state, features, embeddings):
        return {code: predict_family(models, _matrix([features], state['columns'][code]), embeddings, [0])
                for code, models in state['heads'].items()}

    def analyze(self, image):
        result = self.base.analyze(image)
        features = dict(result['features'])
        reference_features = dict(features)
        region, group = result['region'], group_of(result['region'])
        if group == 'spine':
            masks = np.asarray(self.anatomy.predict_single_npy_array(image.pixels[None, None].astype(np.float32),
                {'spacing': (999 * max(image.pixel_mm_x, image.pixel_mm), image.pixel_mm, image.pixel_mm_x)})).squeeze()
            extra, angle = spine_features(masks, image.pixels, image.pixel_mm_x, image.pixel_mm)
            features.update(extra)
            features = anatomical_axis_features(features, angle, self.manifest.get('numbered_anatomy_axis_used', False))
            regions = []
            for channel, name in [(7, 'Th12'), (6, 'L1'), (5, 'L2'), (4, 'L3'), (3, 'L4'), (2, 'L5')]:
                for item in mask_regions(masks[channel].astype(np.uint8)):
                    item['name'] = name
                    regions.append(item)
            result['learned_anatomy'] = {'status': 'evaluated', 'method': 'pretrained_tsxr_named_masks',
                'regions': regions, 'roi_proposals': roi_proposals(regions), 'model_sha256': self.anatomy_sha256,
                'coordinate_units': 'original_image_pixels', 'clinical_validation': False, 'affects_decision': True,
                'target_semantics': 'projected anatomical bone contours; no confirmed disc boundaries'}
        canonical = np.ascontiguousarray(image.pixels[:, ::-1]) if region == 'hip_left' else image.pixels
        embeddings = []
        with torch.inference_mode():
            for processor, encoder, kind, size in self.encoders:
                values, _ = image_input(processor, canonical, size, kind)
                vector, _ = representations(encoder, values, kind, 'mean_global')
                embeddings.append(vector.float().cpu().numpy())
        known_fold = None
        if self.held_out_replay:
            by_pixel = self.replay_manifest['decoded_pixel_to_fold'].get(image.pixel_sha256)
            by_study = self.replay_manifest['dicom_study_to_fold'].get(image.study_uid)
            if by_pixel is not None and by_study is not None and by_pixel != by_study:
                raise ValueError('Replay pixel and DICOM study folds conflict')
            known_fold = by_pixel if by_pixel is not None else by_study
            if known_fold is None:
                raise ValueError('Unregistered study in strict held-out replay; use ensemble mode for new studies')
        folds = [known_fold] if known_fold is not None else sorted({f for f, g in self.folds if g == group})
        decisions, probabilities, raw_per_fold = [], [], []
        for fold in folds:
            state, old = self.folds[fold, group], self.reference[fold, group]
            saved = self.decoders[fold, group]
            spec, ranking = saved['decoder'], saved['ranking_blend']
            raw = self.scores(state, features, embeddings)
            scores, quality = decode_scores(spec, state['codes'], np.column_stack([raw[c] for c in state['codes']]), raw['quality'])
            if ranking is not None:
                reference = self.scores(old, reference_features, embeddings)
                _, old_quality = decode_scores(ranking['prior_decoder'], old['codes'],
                    np.column_stack([reference[c] for c in old['codes']]), reference['quality'])
                quality = ranking['current_weight'] * quality + (1 - ranking['current_weight']) * old_quality
            actual = dict(zip(state['codes'], scores[0].tolist()))
            decision = decide(group, float(quality[0]), actual, features, float(np.nextafter(1., np.inf)), spec['thresholds'])
            decisions.append(decision)
            probabilities.append(float(quality[0]))
            raw_per_fold.append(actual)
        if len(decisions) == 1:
            decision = decisions[0]
            reported_scores = raw_per_fold[0]
        else:
            # Ensemble votes by type; the measured axis is identical in every
            # member. This ensemble is not scored on its training studies.
            codes = list(raw_per_fold[0])
            votes = {c: float(np.mean([c in d['violations'] for d in decisions])) for c in codes}
            decision = decide(group, float(np.mean(probabilities)), votes, features,
                              float(np.nextafter(1., np.inf)), {c: .5 for c in codes})
            reported_scores = votes
        result.update(decision)
        result['criteria'] = reported_scores
        result['research_ensemble'] = {'members': len(decisions), 'criterion_score_basis':
            'fold_violation_vote' if len(decisions) > 1 else 'single_held_out_fold_score',
            'mean_raw_scores': {c: float(np.mean([s[c] for s in raw_per_fold])) for c in raw_per_fold[0]}}
        result['features'] = features
        result['decision_version'] = 'research-live-anatomy-qc-1'
        result['workflow_profile_id'] = self.workflow_profile['profile_id']
        if self.held_out_replay:
            self.records.append({'decoded_pixel_sha256': image.pixel_sha256, 'study': image.study_uid,
                             'region': region, 'folds': folds, 'score': result['score'], 'quality': result['quality'],
                             'violations': result['violations'], 'criterion_scores': result['criteria'],
                             'criterion_states': result['criterion_states'], 'criterion_thresholds': result['criterion_thresholds']})
        return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('input', 'output', 'bank', 'ranking_bank', 'decoders', 'feature_spec', 'anatomy_model', 'baseline_profile'):
        parser.add_argument('--' + name.replace('_', '-'), type=Path, required=True)
    parser.add_argument('--encoders', type=Path, nargs=2, required=True)
    parser.add_argument('--held-out-replay', action='store_true')
    parser.add_argument('--replay-manifest', type=Path)
    parser.add_argument('--require-network-isolation', action='store_true')
    args = parser.parse_args()
    if args.held_out_replay and args.replay_manifest is None:
        parser.error('--held-out-replay requires --replay-manifest')
    network_interfaces = [name for _, name in socket.if_nameindex()]
    if args.require_network_isolation and set(network_interfaces) != {'lo'}:
        parser.error('--require-network-isolation needs a network namespace with loopback only')
    warnings.filterwarnings('ignore', message='Invalid value for VR UI:.*')
    warnings.filterwarnings('ignore', message='TypedStorage is deprecated.*')
    os.environ['DXAQC_WORKFLOW_PROFILE'] = str(args.baseline_profile.resolve())
    candidate = LiveAnatomyCandidate(args.bank, args.ranking_bank, args.decoders, args.feature_spec,
                                    args.anatomy_model, args.encoders, held_out_replay=args.held_out_replay,
                                    replay_manifest=args.replay_manifest)
    summary = run_batch(args.input, args.output, Options(keep_explanation_dir=True), candidate,
                        progress=lambda i, n: print(f'{i}/{n}', flush=True) if i % 25 == 0 else None)
    (args.output / 'raw_candidate_predictions.json').write_text(json.dumps(candidate.records, indent=2) + '\n')
    (args.output / 'executed_live_anatomy_candidate.py').write_bytes(EXECUTION_SOURCE)
    (args.output / 'candidate_protocol.json').write_text(json.dumps({'profile': candidate.workflow_profile,
        'bank_sha256': candidate.manifest['bank_sha256'], 'code_sha256': EXECUTION_SHA256,
        'raw_feature_cache_used': False, 'target_labels_used_by_inference': False,
        'network_interfaces': network_interfaces,
        'replay_manifest_sha256': digest(args.replay_manifest) if args.replay_manifest else None,
        'clinical_validation': False, 'automatic_release_promotions': 0}, indent=2) + '\n')
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
