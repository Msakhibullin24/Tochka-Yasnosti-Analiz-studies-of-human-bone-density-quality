import csv
import json
from types import SimpleNamespace

import numpy as np
import pytest

from conftest import synthetic_spine, write_dicom
from dxaqc.anatomy import evaluate_source_roi
from dxaqc.decision import decide
from dxaqc.pipeline import Analyzer, Options, run_batch
from dxaqc.requirements import evaluate_requirements, image_checks, timing_report


def test_binary_signal_survives_typification_failure_in_all_exports(tmp_path, monkeypatch, bundle_available, trusted_synthetic_router):
    original = Analyzer.analyze
    def untyped(self, img):
        result = original(self, img)
        result.update(decide('spine', .9, {'spine_artifact': .1}, result['features'], .5, {'spine_artifact': .5}))
        return result
    monkeypatch.setattr(Analyzer, 'analyze', untyped)
    src = tmp_path / 'in'; src.mkdir()
    write_dicom(src / 'a.dcm', synthetic_spine(), study_uid='1.2', sop_uid='1.3')
    summary = run_batch(src, tmp_path / 'out', Options(explanations=True))
    for name in ('results.csv', 'results_extended.csv', 'submission.csv'):
        with (tmp_path / 'out' / name).open(encoding='utf-8-sig') as stream:
            row = next(csv.DictReader(stream))
        assert row['quality_class'] == '1' and row['violation_type'] == ''
    assert not summary['submission_valid'] and summary['untyped_violations'] == 1
    assert not summary['requirements_complete']


def test_preview_failure_cannot_remain_a_success(tmp_path, monkeypatch, bundle_available, trusted_synthetic_router):
    import cv2
    monkeypatch.setattr(cv2, 'imwrite', lambda *args: False)
    src = tmp_path / 'in'; src.mkdir()
    write_dicom(src / 'a.dcm', synthetic_spine(), study_uid='1.2', sop_uid='1.3')
    summary = run_batch(src, tmp_path / 'out', Options(explanations=False, keep_explanation_dir=True))
    with (tmp_path / 'out/results.csv').open(encoding='utf-8-sig') as stream:
        row = next(csv.DictReader(stream))
    assert summary['failure'] == 1 and row['processing_status'] == 'Failure'
    assert row['quality_class'] == row['quality_prob'] == row['violation_type'] == ''


def test_single_image_internal_error_has_a_structured_failure(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient
    from dxaqc import api
    def unavailable(_):
        raise RuntimeError('fixture model failure')
    monkeypatch.setattr(api, 'analyzer', lambda: SimpleNamespace(analyze=unavailable))
    path = write_dicom(tmp_path / 'a.dcm', synthetic_spine(), study_uid='1.2', sop_uid='1.3')
    response = TestClient(api.app).post('/api/v1/analyze', files={'file': ('a.dcm', path.read_bytes())})
    assert response.status_code == 500
    assert response.json()['processing_status'] == 'Failure'
    assert response.json()['error_code'] == 'INTERNAL_ERROR'


def test_new_expert_reviews_use_only_official_hip_criteria():
    from dxaqc.api import catalog
    from dxaqc.review import Review
    assert set(catalog()['regions']['hip_left']) == {'hip_position_rotation', 'hip_roi_coverage'}
    review = Review(expected_revision=0, author='fixture', status='confirmed', quality_class=1,
                    violations=['hip_metal_implant'])
    with pytest.raises(ValueError, match='anatomical region'):
        review.for_region('hip_left')


def test_timing_includes_preparation_and_publication_overhead():
    rows = [{'study_uid': 'a', 'time_of_processing': 1.},
            {'study_uid': 'a', 'time_of_processing': 2.},
            {'study_uid': 'b', 'time_of_processing': 3.}]
    timing = timing_report(rows, 16.)
    assert timing['shared_overhead_seconds'] == 10.
    assert timing['study_upper_bound_seconds'] == {'a': 13., 'b': 13.}
    assert timing['within_180_seconds']
    assert not timing_report(rows, 190.)['within_180_seconds']
    with pytest.raises(ValueError): timing_report(rows, float('nan'))


def test_validator_rejects_mismatched_timing_and_missing_typification(tmp_path):
    from dxaqc.report import write_csv
    from dxaqc.validate_results import validate
    row = {'study_uid': 'a', 'image_uid': 'b', 'path_to_study': 'a', 'path_to_file': 'a.dcm',
           'quality_class': 1, 'quality_prob': .8, 'violation_type': '',
           'anatomical_region': 'Поясничный отдел позвоночника',
           'processing_status': 'Success', 'time_of_processing': 1.}
    path = tmp_path / 'results.csv'; write_csv([row], path, True)
    timing = timing_report([row], 10.)
    timing_path = tmp_path / 'timing.json'; timing_path.write_text(json.dumps(timing))
    assert not validate(path, competition=True, timing=timing_path)['valid']
    row['violation_type'] = 'Присутствуют посторонние предметы'
    write_csv([row], path, True)
    assert validate(path, competition=True, timing=timing_path)['valid']
    timing['study_upper_bound_seconds']['a'] = 1.
    timing_path.write_text(json.dumps(timing))
    assert not validate(path, competition=True, timing=timing_path)['valid']


def test_candidate_landmarks_and_absent_roi_cannot_claim_completion():
    img = SimpleNamespace(pixel_mm_source='PixelSpacing')
    result = {'region': 'hip_right', 'features': {'troch_top_margin_mm': 40.,
              'ischium_bottom_margin_mm': 40., 'lateral_margin_mm': 25.},
              'anatomy_candidates': {'landmarks': []}}
    checks = image_checks(img, result, {'value': 'frontal', 'clinical_validation': False}, {'status': 'absent'})
    coverage = next(c for c in checks if c['id'] == 'scan_coverage')
    assert coverage['candidate_result'] == 'pass' and coverage['status'] == 'undetermined'
    assert next(c for c in checks if c['id'] == 'source_roi')['status'] == 'not_applicable'
    row = {'processing_status': 'Success', 'quality_class': 0, 'anatomical_checks_complete': 'false',
           'requirement_checks': json.dumps(checks)}
    assert not evaluate_requirements([row], timing_report([row], 1.))['complete']
    result['features']['troch_top_margin_mm'] = np.nan
    json.dumps(image_checks(img, result, {}, {'status': 'absent'}), allow_nan=False)


def test_neck_roi_is_not_rejected_using_scan_coverage_margins():
    roi = {'id': 'neck', 'purpose': 'femoral_neck', 'bounds': [1, 1, 10, 10],
           'contours': [{'points': [[1, 1], [10, 1], [10, 10], [1, 10]], 'depth': 0}]}
    img = SimpleNamespace(pixels=np.zeros((100, 100)), pixel_mm=1., pixel_mm_x=1., pixel_mm_source='PixelSpacing')
    landmarks = {'landmarks': [{'name': 'femoral_neck', 'points': [[5, 5]]},
                               {'name': 'greater_trochanter', 'points': [[20, 20]]}]}
    check = evaluate_source_roi({'status': 'extracted', 'rois': [roi], 'issues': []}, landmarks, img, 'hip_right')['checks'][0]
    assert check['margin_status'] == 'not_applicable'
    assert check['anatomical_validity'] == 'undetermined'
    assert check['anatomical_relationships'][0]['candidate_inside'] is True
    roi['purpose'] = 'scan_coverage'
    check = evaluate_source_roi({'status': 'extracted', 'rois': [roi], 'issues': []}, landmarks, img, 'hip_right')['checks'][0]
    assert check['margin_status'] == 'fail'


def test_learned_numbering_candidates_are_reported_without_verified_pass():
    from types import SimpleNamespace
    img = SimpleNamespace(pixel_mm_source='PixelSpacing')
    result = {'region':'spine','anatomy_candidates':{'landmarks':[]},
              'learned_anatomy':{'status':'evaluated','regions':[{'name':'L1'},{'name':'L3'}],
                                'roi_proposals':[{'purpose':'L1','requires_confirmation':True}]}}
    checks = image_checks(img,result,{}, {'status':'absent'})
    numbering = next(c for c in checks if c['id']=='vertebral_numbering_and_roi')
    assert numbering['status'] == 'undetermined'
    assert numbering['basis'] == 'learned_numbered_mask_candidates'
    assert numbering['candidate_levels'] == ['L1','L3']
    assert numbering['missing_candidate_levels'] == ['L2','L4']
    assert numbering['disc_boundaries_confirmed'] is False
