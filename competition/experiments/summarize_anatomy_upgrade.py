"""Aggregate measured anatomy/QC and container results without promoting a model."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re


def read(path):
    return json.loads(path.read_text())


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_summary(path):
    if not path.exists():
        return None
    matches = re.findall(r'(\d+) passed, (\d+) skipped', path.read_text())
    return {'passed': int(matches[-1][0]), 'skipped': int(matches[-1][1]),
            'log': str(path), 'log_sha256': digest(path)} if matches else {'finished_successfully': False, 'log': str(path)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs', type=Path, default=Path('data/gpu-runs/anatomy-upgrade-2026-10-15'))
    parser.add_argument('--output', type=Path, default=Path('docs/competition/anatomy_upgrade_2026_10_15.json'))
    args = parser.parse_args()
    trials = []
    baseline = None
    for path in sorted(args.runs.rglob('evaluation.json')):
        source = read(path)
        if 'metrics' not in source or 'screen' not in source:
            continue
        if path.parent.name.startswith('raw-'):
            continue
        screen = source['screen']
        baseline = baseline or screen['overall_score_metrics']['baseline']
        scripts = list(path.parent.glob('executed_*.py')) + list(path.parent.parent.glob('executed_*.py'))
        trials.append({'name': str(path.parent.relative_to(args.runs)), 'metrics': source['metrics'],
                       'paired_delta_f1_ci95': screen['paired_delta_f1_ci95'],
                       'internal_screen_passed': screen['passes_internal_metric_screen'],
                       'internal_screen_reasons': screen['failure_reasons'],
                       'selection_audit': source.get('selection_audit'),
                       'evaluation_sha256': digest(path), 'oof_sha256': digest(path.parent / 'candidate_oof.csv'),
                       'code_sha256_reported': source['code_sha256'],
                       'execution_source_archived_and_hash_verified': any(digest(script) == source['code_sha256'] for script in scripts),
                       'bank_sha256': source.get('bank_sha256'), 'ranking_bank_sha256': source.get('ranking_bank_sha256'),
                       'input_candidate_sha256': source.get('input_candidate_sha256')})
    anatomy = read(args.runs / 'combined-anatomy-v4/features.json')
    spine = [v['features'] for v in anatomy['entries'].values() if 'anat_lumbar_ordered' in v['features']]
    hip = [v['features'] for v in anatomy['entries'].values() if 'anat_femur_present' in v['features']]
    landmarks = {}
    for name in ('vsd-heatmaps', 'vsd-heatmaps-normalized'):
        result = read(args.runs / name / 'evaluation.json')
        result.pop('test_predictions', None)
        landmarks[name] = result
    report = {'date': '2026-10-15', 'schema_version': 1,
              'requirements_complete': False, 'clinical_validation': False, 'model_promotions': 0,
              'release': '1.11.6', 'qc_trials_completed': len(trials), 'baseline': baseline, 'trials': trials,
              'protocol': {'labelled_images': 249, 'study_groups': 100, 'patient_disjoint_established': False,
                           'outer_study_folds': 5, 'inner_study_folds': 4,
                           'conditional_on_historical_routing': True,
                           'previously_inspected_cohort': True,
                           'global_metric_maximum_established': False},
              'anatomy': {'spine_images': len(spine), 'hip_images': len(hip),
                          'candidate_counts': {k: sum(v.get(k) == 1 for v in spine) for k in (
                              'anat_lumbar_ordered', 'anat_Th12_present', 'anat_iliac_left_present', 'anat_iliac_right_present')},
                          'hip_landmark_inputs_available': sum(v.get('anat_landmark_input_valid') == 1 for v in hip),
                          'hip_numbered_axis_geometrically_non_degenerate': sum(v.get('anat_neck_axis_valid') == 1 for v in hip),
                          'feature_columns_by_group': anatomy['columns_by_group'],
                          'features_sha256': digest(args.runs / 'combined-anatomy-v4/features.json'),
                          'segmentation_checkpoints': {k: anatomy[k] for k in ('checkpoint_sha256', 'hip_checkpoint_sha256')},
                          'weight_license': 'CC-BY-NC-4.0',
                          'model_source': 'https://zenodo.org/records/17052912',
                          'candidate_presence_does_not_establish_accuracy': True},
              'synthetic_landmarks': landmarks,
              'public_hip_transfer': read(args.runs / 'public-hip-transfer/evaluation.json'),
              'verification': {'local_tests': test_summary(Path('data/logs/competition-tests-anatomy-final-2026-10-15.log')),
                               'container_tests': test_summary(Path('data/logs/rootless-container-tests-fixed-2026-10-15.log'))},
              'limitations': ['Source anatomy and QC targets differ; candidate detection counts are not anatomical accuracy.',
                              'Some early research runs reported source-file digests at completion; immutable execution-source archives are explicitly audited per run.',
                              'Final container validation uses frozen workflow 1.11.6, not the experimental anatomical QC heads.',
                              'Chinese PNG corpus modality is unverified; visually examined samples resemble ordinary radiographs. It was not used for this QC training.',
                              'Clinical and organizer closed-test performance not established.']}
    inspect_path = Path('data/rootless-audit-2026-10-15/image-inspect.json')
    if inspect_path.exists():
        inspected = read(inspect_path)[0]
        report['verification']['container_image'] = {'id': inspected['Id'], 'created': inspected['Created'],
                                                     'image_source_inspection_sha256': digest(inspect_path)}
    cli = Path('data/rootless-audit-2026-10-15/cli')
    if (cli / 'summary.json').exists():
        summary = read(cli / 'summary.json')
        timing = read(cli / 'timing.json')
        report['verification']['offline_cli'] = {
            'files': summary['files'], 'success': summary['success'], 'failure': summary['failure'],
            'submission_valid': summary['submission_valid'], 'requirements_complete': summary['requirements_complete'],
            'batch_seconds': timing['batch_seconds'], 'max_study_upper_bound_seconds': timing['max_study_upper_bound_seconds'],
            'within_180_seconds': timing['within_180_seconds'], 'network_mode': 'none',
            'submission_csv_sha256': digest(cli / 'submission.csv')}
    http = Path('data/rootless-audit-2026-10-15/http-acceptance.json')
    if http.exists():
        report['verification']['offline_http'] = read(http)
    raw = args.runs / 'raw-live-evaluation-v2/evaluation.json'
    if raw.exists():
        report['verification']['raw_dicom_held_out_replay'] = read(raw)
        report['verification']['raw_dicom_held_out_replay_sha256'] = digest(raw)
        report['verification']['raw_dicom_protocol'] = read(args.runs / 'raw-live-replay-v2/candidate_protocol.json')
    for name in ('camera-projection', 'camera-transfer', 'public-spine-density', 'native-neck-adaptation',
                 'app-heatmaps-1', 'app-heatmaps-2', 'app-heatmaps-noise', 'public-context-hip',
                 'native-context-neck-adaptation', 'final-gated-decoder'):
        path = args.runs / name / 'evaluation.json'
        if path.exists():
            value = read(path)
            value.pop('held_predictions', None)
            value.pop('results', None)
            value.pop('cases', None)
            report[name.replace('-', '_')] = value
    numbered = args.runs / 'raw-numbered-evaluation/evaluation.json'
    if numbered.exists():
        report['verification']['raw_numbered_visibility_replay'] = read(numbered)
        report['verification']['raw_numbered_visibility_summary'] = read(args.runs / 'raw-numbered-visibility/summary.json')
    gated = args.runs / 'raw-own-roi-evaluation/evaluation.json'
    if gated.exists():
        report['verification']['raw_own_roi_gated_replay'] = read(gated)
        report['verification']['raw_own_roi_gated_summary'] = read(args.runs / 'raw-own-roi-gated/summary.json')
    gpu = Path('data/gpu-container-audit-2026-10-15')
    if (gpu / 'image-inspect.json').exists():
        inspected = read(gpu / 'image-inspect.json')[0]
        gpu_timing = read(gpu / 'cli/timing.json')
        gpu_timing = {key: value for key, value in gpu_timing.items() if not isinstance(value, dict)}
        report['verification']['research_gpu_container'] = {
            'id': inspected['Id'], 'bytes': inspected['Size'], 'scope': 'five-fold deployment ensemble; no held-out accuracy claim',
            'inference_source_sha256': read(gpu / 'inference-source-sha256.json'),
            'cli_summary': read(gpu / 'cli/summary.json'), 'cli_timing': gpu_timing,
            'cli_timing_sha256': digest(gpu / 'cli/timing.json'),
            'deployment_protocol': read(gpu / 'cli/deployment_protocol.json')}
        if (gpu / 'http-acceptance.json').exists():
            report['verification']['research_gpu_container']['http'] = read(gpu / 'http-acceptance.json')
    artifact = Path('data/build-artifacts/2026-10-15/SHA256SUMS')
    if artifact.exists():
        report['verification']['portable_images_archive'] = artifact.read_text().strip()
    archive_audit = Path('data/build-artifacts/2026-10-15/archive-verification.json')
    if archive_audit.exists():
        report['verification']['portable_images_archive_verification'] = read(archive_audit)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    print(json.dumps({'qc_trials_completed': len(trials), 'output': str(args.output)}))


if __name__ == '__main__':
    main()
