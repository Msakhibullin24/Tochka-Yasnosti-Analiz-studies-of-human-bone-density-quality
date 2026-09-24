"""Freeze or verify the exact local DXA QC baseline and its evidence.

The snapshot records hashes, not training images or patient identifiers.
It describes the measured 1.9.2 core and the separate 1.9.3 shadow portfolio.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CORE_FILES = (
    'competition/models/bundle.joblib',
    'competition/labels/image_labels.csv',
    'competition/reports/typed_v4_pipeline_oof.csv',
    'competition/reports/typed_v4_metrics.json',
    'competition/reports/release_1_9_2_2026_09_24.json',
    'competition/reports/release_1_9_3_full_sc_sr_2026_09_24.json',
    'competition/reports/release_1_9_3_final_image_2026_09_24.json',
    'competition/reports/typed_v4_container_submission_validation.json',
    'competition/reports/typed_v4_container_extended_validation.json',
)
PORTFOLIO_FILES = (
    'data/specialists/runs/independent-portfolio-v1/portfolio.json',
    'data/specialists/exports/independent-portfolio-v1.tar',
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def build(root: Path, source_audit: Path) -> dict:
    audit = json.loads(source_audit.read_text())
    metrics = json.loads((root / 'competition/reports/typed_v4_metrics.json').read_text())
    release = json.loads((root / 'competition/reports/release_1_9_2_2026_09_24.json').read_text())
    full_series = json.loads((root / 'competition/reports/release_1_9_3_final_image_2026_09_24.json').read_text())
    strict = json.loads((root / 'competition/reports/typed_v4_container_submission_validation.json').read_text())
    portfolio = json.loads((root / PORTFOLIO_FILES[0]).read_text())
    if release['release_version'] != '1.9.2' or release['decision_version'] != '4':
        raise ValueError('Unexpected core release or decision version')
    if not strict['valid'] or strict['images'] != 499 or strict['success'] != 499:
        raise ValueError('Full container strict validation is not 499/499 valid')
    if not release['manifest_and_repeat_validation']['valid']:
        raise ValueError('Full batch manifest or repeat failed')
    if (full_series['derived_sc'], full_series['derived_sr']) != (499, 499) or not (
            full_series['strict_submission']['valid'] and
            full_series['independent_extended_and_series']['valid'] and
            full_series['repeat_predictions_match']):
        raise ValueError('Full SC/SR batch did not pass independent validation')
    if portfolio['mode'] != 'shadow' or portfolio['affects_decision'] is not False:
        raise ValueError('Portfolio cannot be represented as an independent shadow')
    if set(portfolio['members']) != {'spine_convnext', 'hip_convnext', 'general_efficientnet'}:
        raise ValueError('Unexpected specialist portfolio members')
    if len(audit['sources']) != 1 or audit['sources'][0]['name'] != 'organiser':
        raise ValueError('Source audit must describe organiser labels only')
    source_labels = audit['sources'][0]
    labels_hash = sha256(root / 'competition/labels/image_labels.csv')
    oof_hash = sha256(root / 'competition/reports/typed_v4_pipeline_oof.csv')
    if source_labels['labels_sha256'] != labels_hash or metrics['labels_sha256'] != labels_hash:
        raise ValueError('Label checksum differs across source audit and OOF metrics')
    if metrics['predictions_sha256'] != oof_hash:
        raise ValueError('OOF prediction checksum differs from metric report')
    if release['model_bundle_sha256'] != sha256(root / 'competition/models/bundle.joblib'):
        raise ValueError('Core model bundle differs from measured release')
    if release['decision_code_sha256'] != sha256(root / 'competition/dxaqc/decision.py'):
        raise ValueError('Core decision code differs from measured release')
    if (audit['labelled_images'], audit['study_groups'], audit['current_unique_pixels']) != (249, 100, 249):
        raise ValueError('Organiser source audit no longer matches the baseline cohort')
    files = [*CORE_FILES, *PORTFOLIO_FILES]
    files += [str(path.relative_to(root)) for path in sorted((root / 'competition/dxaqc').glob('*.py'))]
    files += ['competition/Dockerfile', 'competition/run.sh',
              'competition/requirements-lock.txt', 'competition/requirements-specialists-lock.txt',
              'competition/train.py', 'competition/train_specialist.py',
              'competition/source_integrity.py', 'competition/ingest.py',
              'competition/pack_independent_specialists.py']
    hashes = {name: {'sha256': sha256(root / name), 'bytes': (root / name).stat().st_size}
              for name in files}
    head = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip()
    dirty = bool(subprocess.check_output(['git', 'status', '--porcelain'], cwd=root, text=True).strip())
    return {
        'schema_version': 1, 'core_version': '1.9.2', 'portfolio_version': '1.9.3',
        'portfolio_mode': 'shadow', 'git_head': head, 'working_tree_dirty_at_capture': dirty,
        'reproducibility_note': 'Hashes identify this working-tree snapshot; git HEAD alone is insufficient when dirty.',
        'source_audit_sha256': sha256(source_audit),
        'source_summary': {key: audit[key] for key in ('labelled_images', 'study_groups',
                            'current_unique_pixels', 'within_study_pixel_copies',
                            'historical_hash_changed', 'patient_disjointness_proven')},
        'measured_core': {
            'oof_images': metrics['labelled_images'], 'oof_studies': metrics['studies'],
            'f1': metrics['overall_quality']['f1'], 'roc_auc': metrics['overall_quality']['roc_auc'],
            'criterion_macro_f1': metrics['criterion_macro_f1'],
            'container_images': strict['images'], 'container_success': strict['success'],
            'strict_valid': strict['valid'],
            'max_seconds_per_study_without_sc_sr': release['container_run']['max_seconds_per_study'],
            'full_sc_sr_images': full_series['summary']['files'],
            'max_seconds_per_study_with_sc_sr': full_series['summary']['max_seconds_per_study'],
        },
        'files': hashes,
        'limits': ['OOF is internal study-held-out evaluation, not independent clinical validation.',
                   'Patient-disjointness is unproven.',
                   'Full batch with SC/SR was measured on the laptop, not on RTX 5090.'],
    }


def comparable(snapshot: dict) -> dict:
    """Capture metadata may change without changing the release artifacts."""
    return {key: value for key, value in snapshot.items()
            if key not in ('git_head', 'working_tree_dirty_at_capture')}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-audit', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--verify', action='store_true', help='Check an existing snapshot without changing it')
    args = parser.parse_args()
    snapshot = build(ROOT, args.source_audit)
    if args.verify:
        recorded = json.loads(args.output.read_text())
        if comparable(recorded) != comparable(snapshot):
            raise SystemExit('Release snapshot differs from current files or evidence')
        print('Release snapshot verified')
        return
    if args.output.exists():
        parser.error('Output exists; choose a new snapshot path or use --verify')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(snapshot, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({key: snapshot[key] for key in ('core_version', 'portfolio_version',
                      'working_tree_dirty_at_capture', 'source_summary', 'measured_core')}, ensure_ascii=False))


if __name__ == '__main__':
    main()
