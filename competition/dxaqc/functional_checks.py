"""Coverage of implemented automatic functions, separate from anatomical proof.

A completed computation may report an uncertain finding. Completion never sets
clinical_validation or changes a medical quality class. Legacy verified-evidence
coverage remains separately available in requirements.json.
"""
from __future__ import annotations
import math
import re


def finite(value):
    return isinstance(value, (int, float)) and math.isfinite(value)


def model_function_checks(img, result, projection, source_roi):
    learned = result.get('learned_anatomy') or {}
    geometry = learned.get('computed_geometry')
    if not isinstance(geometry, dict) or not isinstance(geometry.get('provider_code_sha256'), str):
        return None
    if re.fullmatch('[0-9a-f]{64}', geometry['provider_code_sha256']) is None:
        return None
    group = 'spine' if result['region'] == 'spine' else 'hip'
    scores = result.get('criteria') or {}
    states = result.get('criterion_states') or {}
    checks = []

    def add(name, completed, outcome, basis, evidence=None, resolved=None):
        if resolved is None:
            resolved = outcome not in ('unknown', 'uncertain', 'not_detected', 'unavailable', 'body_not_resolved',
                                        'boundary_ambiguous', 'not_detected_at_both_ends', 'deferred')
        checks.append({'id': name, 'execution_status': 'completed' if completed else 'unavailable',
            'outcome': outcome, 'finding_resolved': bool(completed and resolved), 'basis': basis,
            'review_required': not (completed and resolved) or outcome in ('estimated_fail', 'fail', 'lateral'),
            'evidence': evidence, 'clinical_validation': False,
            'scope': 'model-based automatic assessment; not verified anatomical truth'})

    view = projection.get('value', 'unknown')
    add('projection', bool(projection.get('method')) and view in ('frontal', 'lateral', 'unknown'),
        view, projection.get('method'), {'value': view, 'model_sha256': projection.get('model_sha256')}, resolved=view != 'unknown')
    required_codes = ('spine_coverage', 'spine_axis', 'spine_artifact') if group == 'spine' else ('hip_position_rotation', 'hip_roi_coverage')
    for code in required_codes:
        valid = finite(scores.get(code)) and 0 <= scores[code] <= 1 and states.get(code, {}).get('status') in ('pass', 'fail', 'undetermined')
        add('classifier_'+code, valid, states.get(code, {}).get('status', 'unavailable'),
            'fitted_image_and_geometry_classifier' if code != 'spine_axis' else 'physical_axis_rule',
            {'score': scores.get(code), 'decision': states.get(code)}, resolved=states.get(code, {}).get('status') in ('pass', 'fail'))
    if group == 'spine':
        body = geometry.get('th12') or {}
        outcome = body.get('outcome', 'unavailable')
        add('Th12_visible_half', bool(body.get('method')) and body.get('status') in ('evaluated', 'unavailable'),
            outcome, body.get('method'), body)
        crests = {c['id']: c for c in geometry.get('iliac_crests', [])}
        for name in ('iliac_crest_left', 'iliac_crest_right'):
            item = crests.get(name, {})
            add(name, item.get('status') == 'evaluated', item.get('outcome', 'unavailable'), item.get('method'), item)
        names = {r['name'] for r in learned.get('regions', [])}
        levels = [f'L{i}' for i in range(1, 5) if f'L{i}' in names]
        proposals = learned.get('roi_proposals', [])
        purposes = {p.get('purpose') for p in proposals}
        complete = len(levels) == 4 and all(name in purposes for name in levels)
        add('vertebral_numbering_and_roi', learned.get('status') == 'evaluated' and 'density_based_roi' in learned,
            'proposed' if complete else 'not_detected', 'named_masks_and_density_minima_with_explicit_contour_fallback',
            {'levels': levels, 'roi_proposals': proposals,
             'markers': learned.get('density_based_roi', {}).get('markers', []), 'disc_segmentation_verified': False}, resolved=complete)
    else:
        measurements = learned.get('physical_measurements') or {}
        candidates = {c['name']: c for c in (result.get('anatomy_candidates') or {}).get('landmarks', [])}
        mask_completed = learned.get('status') == 'evaluated' and bool(learned.get('model_sha256'))
        for name in ('greater_trochanter', 'femoral_neck', 'ischium', 'lesser_trochanter'):
            landmark = candidates.get(name, {})
            points = landmark.get('points', [])
            localized = bool(points) and all(len(p) == 2 and all(finite(v) for v in p) for p in points)
            add(name, mask_completed, 'localized' if localized else 'not_detected',
                landmark.get('method', 'named_mask_localization_attempt'), {'points': points, 'model_sha256': learned.get('model_sha256')}, resolved=localized)
        limits = {'anat_troch_top_margin_mm': 30., 'anat_ischium_bottom_margin_mm': 30., 'anat_lateral_margin_mm': 20.}
        margins = {k: measurements.get(k) for k in limits}
        measured = all(finite(v) for v in margins.values())
        scale_known = img.pixel_mm_source not in ('device_default', '', None)
        outcome = 'estimated_pass' if measured and scale_known and all(margins[k] >= limit for k, limit in limits.items()) else 'estimated_fail' if measured and scale_known else 'unavailable'
        add('scan_coverage_mm', mask_completed, outcome, 'predicted_landmarks_and_calibrated_scan_field',
            {'margins_mm': margins, 'minimum_mm': limits, 'scale_source': img.pixel_mm_source}, resolved=measured and scale_known)
        add('rotation_lesser_trochanter', mask_completed and finite(scores.get('hip_position_rotation')),
            states.get('hip_position_rotation', {}).get('status', 'unavailable'), 'combined_position_rotation_model_with_contour_measurements',
            {'lesser_protrusion_mm': measurements.get('anat_lesser_protrusion_mm'),
             'separate_rotation_subtype_ground_truth_available': False,
             'combined_criterion_state': states.get('hip_position_rotation')},
            resolved=finite(measurements.get('anat_lesser_protrusion_mm')))
        roi = learned.get('neck_roi_assessment') or {}
        measured_roi = geometry.get('status') == 'evaluated'
        roi_measures = (learned.get('own_roi_geometry') or {}).get('features') or {}
        alignment_error = roi_measures.get('anat_roi_perpendicular_error_deg')
        add('neck_roi_orientation', mask_completed and (finite(alignment_error) or not measured_roi),
            'measured' if finite(alignment_error) else 'deferred', 'physical_ROI_edges_relative_to_predicted_neck_axis',
            {'perpendicular_error_degrees': alignment_error, 'clinical_angular_tolerance_specified': False},
            resolved=finite(alignment_error))
        add('own_neck_roi_relationships', mask_completed and geometry.get('status') in ('evaluated', 'unavailable'),
            'evaluated' if measured_roi else 'deferred', 'predicted_GT_compartment_named_muscle_tissue_and_pelvic_relationships',
            geometry, resolved=measured_roi)
        if measured_roi:
            gt = geometry.get('greater_trochanter_compartment') or {}
            soft = geometry.get('soft_tissue_sides') or {}
            gt_valid = (finite(gt.get('roi_overlap_mm2')) and finite(gt.get('area_mm2'))
                        and 0 <= gt['roi_overlap_mm2'] <= gt['area_mm2'])
            add('estimated_GT_compartment_exclusion', gt_valid,
                'estimated_pass' if gt.get('estimated_compartment_excluded') is True else 'estimated_fail' if gt.get('estimated_compartment_excluded') is False else 'unknown',
                gt.get('method'), gt, resolved=gt.get('estimated_compartment_excluded') is not None)
            ends = soft.get('ends')
            soft_valid = (isinstance(ends, list) and len(ends) == 2 and all(
                finite(end.get('roi_pixels')) and end['roi_pixels'] > 0
                and finite(end.get('predicted_muscle_nonbone_pixels'))
                and 0 <= end['predicted_muscle_nonbone_pixels'] <= end['roi_pixels'] for end in ends))
            add('soft_tissue_both_sides', soft_valid,
                soft.get('outcome', 'unknown'), soft.get('method'), soft,
                resolved=soft.get('predicted_soft_tissue_at_both_ends') is True)
            pelvic = geometry.get('pelvic_overlap') or {}
            add('pelvic_intersection', finite(pelvic.get('fraction_of_roi')) and 0 <= pelvic['fraction_of_roi'] <= 1,
                'measured_without_invented_failure_tolerance', 'predicted_pelvic_mask_overlap', pelvic,
                resolved=False)
        else:
            for name in ('estimated_GT_compartment_exclusion', 'soft_tissue_both_sides', 'pelvic_intersection'):
                add(name, mask_completed, 'deferred', 'ROI cannot be constructed before anatomy is localized', {'reason': roi.get('reason')}, resolved=False)
    source_status = source_roi.get('status')
    add('source_roi', source_status == 'absent', 'not_applicable' if source_status == 'absent' else 'unknown',
        'source_ROI_is_not_required_for_unmarked_input', {'source_status': source_status}, resolved=source_status == 'absent')
    return {'schema_version': 1, 'checks': checks,
        'execution_complete': bool(checks) and all(c['execution_status'] == 'completed' for c in checks),
        'findings_resolved': bool(checks) and all(c['finding_resolved'] for c in checks),
        'clinical_validation': False,
        'scope': 'Implemented functions executed and produced explicit outcomes; does not assert all findings are correct or conclusive.'}
