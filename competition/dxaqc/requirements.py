"""Evidence coverage, separate from binary classification and CSV conformance.

No missing measurement or candidate landmark is promoted to a clinical pass.
The batch timing includes shared preparation and publication; a study's upper
bound conservatively charges all shared overhead to that study.
"""
from __future__ import annotations

import json
import math
from collections import Counter, defaultdict

LANDMARKS = {
    'spine': ('Th12', 'iliac_crest_left', 'iliac_crest_right'),
    'hip': ('greater_trochanter', 'femoral_neck', 'ischium', 'lesser_trochanter'),
}


def image_checks(img, result, projection, source_roi):
    anatomy = result['anatomy_candidates']
    points = {item['name']: item for item in anatomy['landmarks']}
    group = 'spine' if result['region'] == 'spine' else 'hip'
    checks = []
    for name in LANDMARKS[group]:
        item = points.get(name, {})
        checks.append({'id': name, 'status': 'evaluated' if item.get('verified') is True else 'undetermined',
                       'basis': item.get('method', 'localizer_unavailable'),
                       'candidate_points': item.get('points', []),
                       'reason': 'Нужна подтверждённая локализация и оценка видимости ориентира.'})
    checks.append({'id': 'projection',
                   'status': 'evaluated' if projection.get('clinical_validation') is True else 'undetermined',
                   'basis': projection.get('method'), 'value': projection.get('value'),
                   'reason': projection.get('reason')})
    if group == 'spine':
        learned = result.get('learned_anatomy', {})
        names = {r['name'] for r in learned.get('regions', [])} if learned.get('status') == 'evaluated' else set()
        levels = [f'L{i}' for i in range(1, 5) if f'L{i}' in names]
        checks.append({'id': 'vertebral_numbering_and_roi', 'status': 'undetermined',
                       'basis': 'learned_numbered_mask_candidates' if levels else 'numbered_vertebrae_and_disc_boundaries_unavailable',
                       'candidate_levels': levels, 'missing_candidate_levels': [f'L{i}' for i in range(1,5) if f'L{i}' not in names],
                       'roi_proposals': learned.get('roi_proposals', []), 'disc_boundaries_confirmed': False,
                       'reason': 'Кандидаты уровней и ROI доступны; нумерация и межпозвоночные границы не подтверждены.'
                                 if levels else 'Нумерация L1–L4 и межпозвоночные границы автоматически не подтверждены.'})
    else:
        required = ('troch_top_margin_mm', 'ischium_bottom_margin_mm', 'lateral_margin_mm')
        values = {k: result['features'].get(k) for k in required}
        values = {k: float(v) if isinstance(v, (int, float)) and math.isfinite(v) else None for k, v in values.items()}
        reliable = img.pixel_mm_source not in ('device_default', '', None)
        finite = all(isinstance(v, (int, float)) and math.isfinite(v) for v in values.values())
        limits = dict(zip(required, (30., 30., 20.)))
        checks.append({'id': 'scan_coverage', 'status': 'undetermined',
                       'basis': 'candidate_landmark_margins', 'margins_mm': values,
                       'limits_mm': limits, 'scale_source': img.pixel_mm_source,
                       'candidate_result': ('pass' if all(values[k] >= limits[k] for k in required) else 'fail')
                                           if finite and reliable else 'unavailable',
                       'reason': 'Отступы относятся к охвату сканирования; ориентиры требуют проверки.'})
        checks.append({'id': 'rotation', 'status': 'undetermined',
                       'basis': 'model_and_candidate_lesser_trochanter',
                       'reason': 'Недостаточная и избыточная ротация требуют подтверждённого контура малого вертела.'})
    checks.append({'id': 'source_roi',
                   'status': 'not_applicable' if source_roi['status'] == 'absent' else 'undetermined',
                   'basis': source_roi['status'],
                   'reason': 'Во входе без ROI невозможно проверить исходную разметку.'
                             if source_roi['status'] == 'absent' else 'Назначение и анатомическая корректность ROI требуют проверки.'})
    return checks


def timing_report(rows, total_seconds):
    if not math.isfinite(total_seconds) or total_seconds < 0:
        raise ValueError('invalid batch duration')
    by_study = defaultdict(float)
    for i, row in enumerate(rows):
        elapsed = float(row.get('time_of_processing', 0))
        if not math.isfinite(elapsed) or elapsed < 0:
            raise ValueError('invalid image duration')
        key = row.get('study_uid') or row.get('path_to_study') or f'row:{i}'
        by_study[key] += elapsed
    overhead = max(0., total_seconds - sum(by_study.values()))
    upper = {key: value + overhead for key, value in by_study.items()}
    return {'version': 1, 'batch_seconds': total_seconds,
            'shared_overhead_seconds': overhead, 'study_image_seconds': dict(by_study),
            'study_upper_bound_seconds': upper,
            'max_study_upper_bound_seconds': max(upper.values(), default=total_seconds),
            'within_180_seconds': max(upper.values(), default=total_seconds) <= 180.,
            'scope': 'local end-to-end batch through tables and series archive; all shared overhead charged to each study; not H200 measurement'}


def _parsed(row, key, default):
    value = row.get(key)
    if isinstance(value, (dict, list)):
        return value
    try:
        return json.loads(value) if value else default
    except (TypeError, ValueError):
        return default


def evaluate_requirements(rows, timing=None):
    successful = [r for r in rows if r.get('processing_status') == 'Success']
    untyped = sum(str(r.get('quality_class')) == '1' and not r.get('violation_type') for r in successful)
    coverage = defaultdict(Counter)
    for row in successful:
        for check in _parsed(row, 'requirement_checks', []):
            coverage[check['id']][check['status']] += 1
    anatomy_complete = bool(successful) and all(
        str(r.get('anatomical_checks_complete')).lower() == 'true' for r in successful)
    failures = len(rows) - len(successful)
    blockers = []
    if not successful:
        blockers.append('no_successful_images')
    if failures:
        blockers.append('processing_failures')
    if untyped:
        blockers.append('violation_type_undetermined')
    if not anatomy_complete:
        blockers.append('anatomical_checks_incomplete')
    if timing is None:
        blockers.append('end_to_end_timing_unavailable')
    elif not timing['within_180_seconds']:
        blockers.append('study_time_upper_bound_exceeded')
    return {'version': 1, 'complete': not blockers, 'images': len(rows),
            'success': len(successful), 'failure': failures, 'untyped_violations': untyped,
            'typification_complete': bool(successful) and not untyped,
            'anatomical_checks_complete': anatomy_complete,
            'checks': {key: dict(value) for key, value in coverage.items()}, 'blockers': blockers,
            'scope': 'coverage of required evidence, not external clinical certification or organiser approval'}
