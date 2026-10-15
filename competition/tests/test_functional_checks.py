import copy
from types import SimpleNamespace

from dxaqc.functional_checks import model_function_checks


def fixture():
    img = SimpleNamespace(pixel_mm_source='organizer_fixed_scanner')
    result = {'region': 'hip_right', 'quality': 0,
        'criteria': {'hip_position_rotation': .2, 'hip_roi_coverage': .3},
        'criterion_states': {'hip_position_rotation': {'status': 'pass'}, 'hip_roi_coverage': {'status': 'pass'}},
        'anatomy_candidates': {'landmarks': [{'name': name, 'points': [[10., 20.]], 'method': 'predicted_mask'}
            for name in ('greater_trochanter', 'femoral_neck', 'lesser_trochanter', 'ischium')]},
        'learned_anatomy': {'status': 'evaluated', 'model_sha256': 'b'*64,
            'physical_measurements': {'anat_troch_top_margin_mm': 35., 'anat_ischium_bottom_margin_mm': 40.,
                                      'anat_lateral_margin_mm': 25., 'anat_lesser_protrusion_mm': 1.},
            'own_roi_geometry': {'features': {'anat_roi_perpendicular_error_deg': 0.}},
            'computed_geometry': {'status': 'evaluated', 'provider_code_sha256': 'a'*64,
                'greater_trochanter_compartment': {'method': 'landmark_partition', 'area_mm2': 100., 'roi_overlap_mm2': 0., 'estimated_compartment_excluded': True},
                'pelvic_overlap': {'fraction_of_roi': 0.},
                'soft_tissue_sides': {'method': 'predicted_muscles', 'ends': [{'roi_pixels': 20, 'predicted_muscle_nonbone_pixels': 0}]*2,
                                     'outcome': 'not_detected_at_both_ends', 'predicted_soft_tissue_at_both_ends': False}}}}
    projection = {'method': 'learned_projection', 'value': 'frontal', 'clinical_validation': False}
    return img, result, projection, {'status': 'absent'}


def test_completed_function_does_not_imply_resolved_soft_tissue_or_clinical_proof():
    args = fixture(); before = copy.deepcopy(args[1])
    checks = model_function_checks(*args)
    assert checks['execution_complete'] is True
    assert checks['findings_resolved'] is False
    assert checks['clinical_validation'] is False
    assert all(c['clinical_validation'] is False for c in checks['checks'])
    assert args[1] == before
    soft = next(c for c in checks['checks'] if c['id'] == 'soft_tissue_both_sides')
    assert soft['execution_status'] == 'completed' and soft['review_required']


def test_clinical_flag_alone_cannot_create_function_coverage():
    args = fixture();args[1]['learned_anatomy'].pop('computed_geometry')
    args[1]['learned_anatomy']['clinical_validation'] = True
    assert model_function_checks(*args) is None


def test_missing_mandatory_classifier_is_not_completed():
    args = fixture();args[1]['criteria'].pop('hip_roi_coverage')
    checks = model_function_checks(*args)
    assert checks['execution_complete'] is False
    failed = next(c for c in checks['checks'] if c['id'] == 'classifier_hip_roi_coverage')
    assert failed['execution_status'] == 'unavailable'


def test_unknown_physical_scale_does_not_make_margin_pass():
    args = fixture();args[0].pixel_mm_source = 'device_default'
    checks = model_function_checks(*args)
    margin = next(c for c in checks['checks'] if c['id'] == 'scan_coverage_mm')
    assert margin['outcome'] == 'unavailable'
    assert margin['finding_resolved'] is False


def test_unlocalized_roi_stays_deferred_without_invented_coordinates():
    args = fixture();learned = args[1]['learned_anatomy']
    learned['computed_geometry'] = {'status': 'unavailable', 'reason': 'landmarks_unavailable', 'provider_code_sha256': 'a'*64}
    learned['neck_roi_assessment'] = {'status': 'unavailable', 'reason': 'landmarks_unavailable'}
    checks = model_function_checks(*args)
    own = next(c for c in checks['checks'] if c['id'] == 'own_neck_roi_relationships')
    assert own['outcome'] == 'deferred'
    assert own['finding_resolved'] is False


def test_batch_exports_preserve_uncertainty_and_legacy_evidence(tmp_path):
    import csv
    import json
    from dxaqc.report import write_csv, CONTRACT_COLUMNS
    from dxaqc.requirements import evaluate_requirements, timing_report
    functions = model_function_checks(*fixture())
    row = {'processing_status': 'Success', 'quality_class': 0, 'time_of_processing': .1,
           'anatomical_checks_complete': 'false',
           'automatic_requirement_checks': json.dumps(functions['checks'])}
    path = tmp_path/'extended.csv'
    write_csv([row], path)
    with path.open(encoding='utf-8-sig') as stream:
        exported = next(csv.DictReader(stream))
    report = evaluate_requirements([exported], timing_report([exported], 1.))
    assert report['automatic_check_execution_complete'] is True
    assert report['automatic_workflow_complete'] is True
    assert report['automatic_findings_resolved'] is False
    assert report['anatomical_checks_complete'] is False
    assert report['complete'] is False
    assert report['automatic_checks']['soft_tissue_both_sides']['review_required'] == 1
    write_csv([row], tmp_path/'strict.csv', True)
    with (tmp_path/'strict.csv').open(encoding='utf-8-sig') as stream:
        assert next(csv.reader(stream)) == CONTRACT_COLUMNS + ['quality_prob']
    # A summary flag alone cannot replace the actual per-function outcomes.
    row['automatic_check_execution_complete'] = 'true'
    row.pop('automatic_requirement_checks')
    assert not evaluate_requirements([row], timing_report([row], 1.))['automatic_workflow_complete']
