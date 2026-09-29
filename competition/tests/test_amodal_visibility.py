"""Unclipped-body geometry never infers the full size from a visible fragment."""
import numpy as np
import pytest
from amodal_visibility import projected_visibility


def test_exact_half_body_uses_pixel_cell_viewport():
    points=[[-.5,-10.5],[9.5,-10.5],[9.5,9.5],[-.5,9.5]]
    result=projected_visibility(points,(10,10))
    assert result['visible_projected_area_fraction']==.5
    assert result['visible_vertical_extent_fraction']==.5
    assert result['clinical_verdict']=='undetermined'
    reversed_result=projected_visibility(points[::-1],(10,10))
    assert reversed_result['visible_projected_area_fraction']==.5


def test_fully_visible_and_absent_bodies():
    inside=[[1,1],[8,1],[8,8],[1,8]]
    assert projected_visibility(inside,(10,10))['visible_projected_area_fraction']==1
    for offset in ([20,0],[0,-20]):
        absent=np.array(inside)+offset
        result=projected_visibility(absent,(10,10))
        assert result['visible_projected_area_fraction']==result['visible_vertical_extent_fraction']==0


def test_half_height_is_not_automatically_half_area():
    result=projected_visibility([[-.5,-10.5],[1.5,-10.5],[9.5,9.5],[-.5,9.5]],(10,10))
    assert result['visible_vertical_extent_fraction']==.5
    assert result['visible_projected_area_fraction']==pytest.approx(2/3)


@pytest.mark.parametrize('points', [[[0,0],[5,5],[0,5],[5,0]],[[0,0],[5,0],[5,0],[0,5]],[[0,0],[5,0],[5,float('nan')],[0,5]]])
def test_invalid_full_body_geometry_rejected(points):
    with pytest.raises(ValueError):projected_visibility(points,(10,10))
