from experiments.anatomy_display_selection import prefer_refinement


def anatomy(roi, geometry):
    return {'neck_roi_assessment': {'status': roi}, 'own_roi_geometry': {'status': geometry}}


def test_secondary_cannot_discard_an_available_roi_when_its_axis_collapses():
    good = anatomy('proposed', 'evaluated')
    collapsed = anatomy('unavailable', 'unavailable')
    assert not prefer_refinement(good, collapsed)
    assert prefer_refinement(collapsed, good)
    assert prefer_refinement(good, good)
    # A proposed polygon alone does not establish computable geometry.
    assert not prefer_refinement(good, anatomy('proposed', 'unavailable'))


def test_spine_contour_fills_only_missing_level_and_never_invents_a_disc():
    from experiments.anatomy_display_selection import complete_spine_roi_proposals
    density = [{'purpose': 'L2', 'method': 'density_minimum'}]
    contours = [{'purpose': name, 'reference_origin': 'learned_mask'} for name in ['L1', 'L2', 'L3', 'L4']]
    chosen = complete_spine_roi_proposals(density, contours)
    assert [item['purpose'] for item in chosen] == ['L1', 'L2', 'L3', 'L4']
    assert chosen[1] is density[0]
    assert chosen[0]['density_boundary_unavailable'] is True
    assert chosen[0]['disc_segmentation_confirmed'] is False
    assert 'density_boundary_unavailable' not in contours[0]
