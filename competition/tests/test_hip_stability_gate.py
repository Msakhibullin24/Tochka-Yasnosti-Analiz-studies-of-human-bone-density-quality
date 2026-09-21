from types import SimpleNamespace

import numpy as np

from dxaqc import anatomy
from dxaqc.geometry import Measurement


def test_stability_gate_suppresses_moved_point_and_keeps_model_verdict():
    base = {'landmarks': [anatomy._candidate('femoral_neck', [(20, 40)], 'test'),
                          anatomy._candidate('ischium', [(50, 100)], 'test')]}
    shifted = {'landmarks': [anatomy._candidate('femoral_neck', [(30, 40)], 'test'),
                             anatomy._candidate('ischium', [(51, 100)], 'test')]}
    result = anatomy.compare_hip_stability(base, shifted,
                                            {'shaft_abs_angle_deg': 2., 'lesser_troch_protrusion_mm': 1.},
                                            {'shaft_abs_angle_deg': 2.2, 'lesser_troch_protrusion_mm': 1.2},
                                            1., 1.)
    assert result['unstable_landmarks'] == ['femoral_neck']
    assert base['landmarks'][0]['points'] == []
    assert base['landmarks'][0]['status'] == 'unstable'
    assert base['landmarks'][1]['points'] == [[50., 100.]]


def test_shared_assessment_adds_review_flag_without_changing_quality(monkeypatch):
    base = {'landmarks': [anatomy._candidate('femoral_neck', [(20, 40)], 'test')]}
    shifted = {'landmarks': [anatomy._candidate('femoral_neck', [(30, 40)], 'test')]}
    outcomes = iter((base, shifted))
    monkeypatch.setattr(anatomy, 'detect_landmarks', lambda *_: next(outcomes))
    monkeypatch.setattr(anatomy, 'measure_image', lambda *_: Measurement(
        {'shaft_abs_angle_deg': 2., 'lesser_troch_protrusion_mm': 1.}))
    img = SimpleNamespace(pixels=np.zeros((150, 120), np.uint8), pixel_mm=1.,
                          pixel_mm_x=1., pixel_mm_source='PixelSpacing', view_position='',
                          source_roi={'status': 'absent', 'rois': [], 'issues': []})
    result = {'region': 'hip_right', 'overlay': {}, 'quality': 0, 'score': .2,
              'features': {'shaft_abs_angle_deg': 2., 'lesser_troch_protrusion_mm': 1.},
              'review_reasons': []}
    assessment = anatomy.image_assessment(img, result)
    assert assessment['anatomy']['landmarks'][0]['points'] == []
    assert assessment['anatomy']['stability']['status'] == 'needs_review'
    assert result['review_reasons'] == ['hip_geometry_unstable']
    assert result['quality'] == 0 and result['score'] == .2
    assert anatomy.image_assessment(img, result)['anatomy'] is assessment['anatomy']
    assert result['review_reasons'] == ['hip_geometry_unstable']


def test_unknown_pixel_scale_does_not_apply_millimetre_gate(monkeypatch):
    base = {'landmarks': [anatomy._candidate('femoral_neck', [(20, 40)], 'test')]}
    monkeypatch.setattr(anatomy, 'detect_landmarks', lambda *_: base)
    img = SimpleNamespace(pixels=np.zeros((150, 120), np.uint8), pixel_mm=.607,
                          pixel_mm_x=.607, pixel_mm_source='device_default', view_position='',
                          source_roi={'status': 'absent', 'rois': [], 'issues': []})
    result = {'region': 'hip_right', 'overlay': {}, 'quality': 0, 'score': .2,
              'features': {}, 'review_reasons': []}
    assessment = anatomy.image_assessment(img, result)
    assert assessment['anatomy']['stability']['status'] == 'unavailable'
    assert assessment['anatomy']['landmarks'][0]['points'] == [[20., 40.]]
    assert result['review_reasons'] == []
