"""Explicit research-candidate API/CLI entrypoint using an inference-only export.

Configuration identifies all model artifacts locally. Deployment banks declare
a full-data fit or fold ensemble; neither is scored on its training images.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import socket
import sys
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def load_candidate(config_path, *, held_out_replay=False, replay_manifest=None):
    config_path = Path(config_path).resolve()
    config = json.loads(config_path.read_text())
    if config.get('schema_version') != 1 or config.get('clinical_validation') is not False:
        raise ValueError('Expected an explicitly unvalidated research candidate configuration')
    def path(name):
        return (config_path.parent / config[name]).resolve()
    from gpu_research.common import digest
    for name in ('hip_landmarks', 'hip_projection', 'neck_template', 'hip_display_landmarks', 'hip_display_template'):
        if config.get(name) and digest(path(name)) != config[name + '_sha256']:
            raise ValueError('Hip checkpoint identity differs from deployment configuration')
    os.environ['DXAQC_WORKFLOW_PROFILE'] = str(path('baseline_profile'))
    from experiments.live_anatomy_candidate import LiveAnatomyCandidate
    candidate_type = LiveAnatomyCandidate
    if config.get('automatic_geometry', False):
        from experiments.automatic_anatomy_candidate import AutomaticAnatomyCandidate
        candidate_type = AutomaticAnatomyCandidate
    return candidate_type(path('bank'), path('ranking_bank'), path('decoders'), path('feature_spec'),
        path('anatomy_model'), [(config_path.parent / p).resolve() for p in config['encoders']],
        pelvis_model=path('pelvis_model'), pelvis_checkpoint_sha256=config['pelvis_checkpoint_sha256'],
        hip_landmarks=path('hip_landmarks') if config.get('hip_landmarks') else None,
        hip_projection=path('hip_projection') if config.get('hip_projection') else None,
        neck_template=path('neck_template') if config.get('neck_template') else None,
        hip_display_landmarks=path('hip_display_landmarks') if config.get('hip_display_landmarks') else None,
        hip_display_template=path('hip_display_template') if config.get('hip_display_template') else None,
        visibility_cue=config.get('visibility_cue', True),
        held_out_replay=held_out_replay, replay_manifest=replay_manifest)


def create_app():
    candidate = load_candidate(os.environ['DXAQC_ANATOMY_CONFIG'])
    from dxaqc import api
    api._analyzer = candidate
    return api.app


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=os.environ.get('DXAQC_ANATOMY_CONFIG'), required='DXAQC_ANATOMY_CONFIG' not in os.environ)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--require-network-isolation', action='store_true')
    parser.add_argument('--held-out-replay', action='store_true')
    parser.add_argument('--replay-manifest', type=Path)
    args = parser.parse_args()
    if args.held_out_replay != bool(args.replay_manifest):
        parser.error('Held-out replay requires both --held-out-replay and --replay-manifest')
    if args.require_network_isolation and {name for _, name in socket.if_nameindex()} != {'lo'}:
        parser.error('Expected loopback-only isolated inference namespace')
    from dxaqc.pipeline import run_batch, Options
    started = perf_counter()
    candidate = load_candidate(args.config, held_out_replay=args.held_out_replay, replay_manifest=args.replay_manifest)
    initialization_seconds = perf_counter() - started
    summary = run_batch(args.input, args.output, Options(keep_explanation_dir=True), candidate)
    if args.held_out_replay:
        (args.output / 'raw_candidate_predictions.json').write_text(json.dumps(candidate.records, indent=2) + '\n')
    if hasattr(candidate, 'geometry_records'):
        (args.output / 'automatic_geometry_report.json').write_text(json.dumps({
            'records': candidate.geometry_records, 'provider_code_sha256': candidate.geometry_code_sha256,
            'clinical_validation': False, 'qc_targets_used': False,
            'scope': 'model-derived measurements; not verified anatomical truth'}, indent=2) + '\n')
    (args.output / 'deployment_protocol.json').write_text(json.dumps({'profile': candidate.workflow_profile,
        'network_interfaces': [name for _, name in socket.if_nameindex()], 'clinical_validation': False,
        'initialization_seconds': initialization_seconds,
        'wall_seconds_including_initialization': perf_counter() - started,
        'cold_study_upper_bound_seconds': summary['max_seconds_per_study_upper_bound'] + initialization_seconds,
        'cold_within_180_seconds': summary['max_seconds_per_study_upper_bound'] + initialization_seconds <= 180,
        'cached_image_features_used': False, 'target_labels_used_by_inference': False}, indent=2) + '\n')
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
