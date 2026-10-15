"""Keep a valid primary ROI when a display-only refinement cannot build one."""


def prefer_refinement(primary, refined):
    def usable(anatomy):
        return (anatomy.get('neck_roi_assessment', {}).get('status') == 'proposed'
                and anatomy.get('own_roi_geometry', {}).get('status') == 'evaluated')
    return usable(refined) or not usable(primary)


def complete_spine_roi_proposals(density_proposals, contour_proposals):
    """Keep density boundaries when available; expose a named contour otherwise.

    A contour fallback supplies no evidence of a located intervertebral disc.
    """
    selected = {item['purpose']: item for item in density_proposals}
    for contour in contour_proposals:
        name = contour['purpose']
        if name in ('L1', 'L2', 'L3', 'L4') and name not in selected:
            selected[name] = {**contour, 'method': 'named_vertebral_contour_fallback',
                              'density_boundary_unavailable': True, 'disc_segmentation_confirmed': False}
    return [selected[name] for name in ('L1', 'L2', 'L3', 'L4') if name in selected]
