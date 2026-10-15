import numpy as np

from experiments.adapt_native_neck_roi import anatomical_frame, proposal_from_template


def test_neck_roi_template_preserves_perpendicular_geometry_with_anisotropic_pixels():
    head, neck = np.array([100., 60.]), np.array([65., 85.])
    sx, sy = .6, 1.05
    center, frame, length = anatomical_frame(head, neck, sx, sy)
    proposal = proposal_from_template([0., 0., .8, 1.4], center, frame, length, sx, sy, (200, 200))
    assert proposal['status'] == 'proposed'
    physical = np.asarray(proposal['polygon']) * [sx, sy]
    along, across = physical[1] - physical[0], physical[2] - physical[1]
    assert abs(along @ across) < 1e-10
    assert np.allclose(np.mean(physical, axis=0), neck * [sx, sy])
    assert np.isclose(np.linalg.norm(along), .8 * length)
    assert np.isclose(np.linalg.norm(across), 1.4 * length)
    outside = proposal_from_template([100., 0., .8, 1.4], center, frame, length, sx, sy, (200, 200))
    assert outside['status'] == 'unavailable'
