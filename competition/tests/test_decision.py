import numpy as np
import pytest
from dxaqc.decision import decide
from dxaqc.geometry import measure_image
from conftest import synthetic_spine


def test_independent_criteria_unknown_type_and_metal_warning():
    result = decide('hip', .2, {'hip_roi_coverage': .8}, {}, .5, {'hip_roi_coverage': .7})
    assert result['quality'] == 1 and result['violations'] == ['hip_roi_coverage']
    fallback = decide('spine', .8, {'spine_artifact': .3}, {}, .5, {'spine_artifact': .7})
    assert fallback['violations'] == [] and fallback['violation_type_status'] == 'undetermined'
    assert fallback['quality'] == 1 and fallback['decision_reason'] == 'binary_only_review'
    result = decide('hip', .2, {'hip_roi_coverage': .1}, {'sat_frac': .02}, .5, {})
    assert result['quality'] == 0 and result['score'] == .2 and not result['violations']
    assert 'suspected_metal_requires_review' in result['review_reasons']


@pytest.mark.parametrize('angle,quality', [(4.99, 0), (5., 0), (5.01, 1), (6.93, 1)])
def test_measured_axis_obeys_five_degree_boundary(angle,quality):
    result=decide('spine',.1,{'spine_axis':.9}, {'spine_abs_angle_deg':angle},.5,{'spine_axis':.5})
    assert result['quality']==quality
    assert ('spine_axis' in result['violations'])==bool(quality)
    assert result['criterion_states']['spine_axis']['basis']=='measured_axis_angle'


def test_unavailable_axis_does_not_become_a_confirmed_type():
    result=decide('spine',.8,{'spine_axis':.9},{'spine_abs_angle_deg':float('nan')},.5,{})
    assert result['violations']==[] and result['criterion_states']['spine_axis']['status']=='undetermined'


def test_anisotropic_physical_angle_and_overlay_coordinates():
    pixels = synthetic_spine(8)
    iso = measure_image(pixels, 'spine', .6, .6)
    wide = measure_image(pixels, 'spine', .6, 1.2)
    assert abs(wide.features['spine_angle_deg']) > abs(iso.features['spine_angle_deg']) * 1.7
    assert all(0 <= x < pixels.shape[1] for x, _ in wide.overlay['axis'])


def test_nan_score_never_becomes_normal():
    with pytest.raises(ValueError):
        decide('spine', np.nan, {}, {}, .5, {})


def test_threshold_can_predict_all_positive_or_all_negative():
    from dxaqc.model import best_f1_threshold
    score = np.array([.1, .2, .3])
    assert np.all(score >= best_f1_threshold(np.ones(3), score))
    assert not np.any(score >= best_f1_threshold(np.zeros(3), score))


def test_pixel_spacing_x_is_preserved(tmp_path):
    import pydicom
    from conftest import write_dicom
    from dxaqc.dicom_io import read_dxa
    path = write_dicom(tmp_path / 'anisotropic.dcm', synthetic_spine(8))
    ds = pydicom.dcmread(path)
    ds.PixelSpacing = [.6, 1.2]
    ds.save_as(path)
    image = read_dxa(path)
    assert image.pixel_mm == .6 and image.pixel_mm_x == 1.2
