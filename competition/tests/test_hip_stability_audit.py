import numpy as np

from experiments.hip_stability_audit import compare, perturb, summarize, VARIANTS, POINTS


def test_comparison_distinguishes_missing_point_from_coordinate_shift():
    base = ({'shaft_abs_angle_deg': 2., 'lesser_troch_protrusion_mm': 4.},
            {name: [[10., 20.]] for name in POINTS})
    changed = ({'shaft_abs_angle_deg': 3., 'lesser_troch_protrusion_mm': 7.},
               {name: ([[12., 23.]] if name != 'femoral_neck' else []) for name in POINTS})
    result = compare(base, changed, 1., .5)
    assert result['shaft_abs_angle_deg_delta'] == 1.
    assert result['lesser_troch_protrusion_mm_delta'] == 3.
    assert result['greater_trochanter_shift_mm'] == np.sqrt(10)
    assert result['femoral_neck_status_flip'] is True
    assert result['femoral_neck_shift_mm'] is None


def test_summary_keeps_error_groups_separate():
    measurements = {name + '_delta': 0. for name in ('shaft_abs_angle_deg',
                                                      'lesser_troch_protrusion_mm')}
    measurements.update({name + '_status_flip': False for name in POINTS})
    measurements.update({name + '_shift_mm': None for name in POINTS})
    cases = [{'group': group, 'variants': {v: measurements for v in VARIANTS}}
             for group in ('fn', 'tn')]
    report = summarize(cases)
    assert report['bright']['all']['images'] == 2
    assert report['bright']['fn']['images'] == 1
    assert report['bright']['fp']['images'] == 0
    assert report['bright']['fn']['greater_trochanter_paired_candidates'] == 0


def test_perturbation_preserves_shape_and_rejects_unknown_variant():
    image = np.full((20, 30), 100, np.uint8)
    assert all(perturb(image, name).shape == image.shape for name in VARIANTS)
    import pytest
    with pytest.raises(ValueError, match='unknown perturbation'):
        perturb(image, 'unknown')
