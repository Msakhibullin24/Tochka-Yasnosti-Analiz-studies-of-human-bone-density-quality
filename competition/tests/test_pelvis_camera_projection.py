import numpy as np
import pytest

from experiments.pelvis_camera_projection import pelvic_frame, project, rotate_about


def pelvis_points():
    points = np.zeros((14, 3))
    points[2], points[3] = [100, 0, 50], [-100, 0, 50]
    points[6], points[7] = [20, 0, -50], [-20, 0, -50]
    return points


def test_anatomical_camera_projection_is_invariant_to_source_ct_rotation_and_translation():
    points = pelvis_points()
    vertices = np.array([[10, 20, 30], [40, 50, 60]])
    rotation = np.array([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]])
    shift = np.array([100, 200, -300])
    frame = pelvic_frame(points)
    transformed = pelvic_frame(points @ rotation.T + shift)
    assert np.allclose(frame @ frame.T, np.eye(3))
    assert np.linalg.det(frame) == pytest.approx(1)
    for angle in (0, 30, 90):
        assert np.allclose(project(vertices, np.zeros(3), frame, angle),
                           project(vertices @ rotation.T + shift, shift, transformed, angle))


def test_leg_rotation_preserves_joint_center_and_longitudinal_axis_without_rotating_pelvis():
    head = np.array([10., 20., 30.])
    axis = np.array([0., 0., 1.])
    points = np.array([head, head + axis * 20, head + [10, 0, 0]])
    rotated = rotate_about(points, head, axis, 90)
    assert np.allclose(rotated[:2], points[:2])
    assert np.allclose(rotated[2], head + [0, 10, 0])
    with pytest.raises(ValueError):
        pelvic_frame(np.zeros((14, 3)))
