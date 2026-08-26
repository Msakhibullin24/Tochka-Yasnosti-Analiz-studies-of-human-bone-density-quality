from app.keypoints import KEYPOINTS, Landmark
from app.qc import assess_landmarks, summarize


def landmark(name: str, x: float, y: float, confidence: float = 0.95) -> Landmark:
    return Landmark(name, x, y, confidence, True)


def complete_landmarks() -> list[Landmark]:
    points = [landmark(name, 0.5, 0.5) for name in KEYPOINTS]
    overrides = {
        "crown": (0.5, 0.02), "groin": (0.5, 0.55), "lower_groin": (0.5, 0.58),
        "right_shoulder": (0.35, 0.2), "left_shoulder": (0.65, 0.2),
        "right_elbow": (0.25, 0.35), "left_elbow": (0.75, 0.35),
        "right_waist": (0.4, 0.45), "left_waist": (0.6, 0.45),
        "right_hip_skin": (0.4, 0.55), "left_hip_skin": (0.6, 0.55),
        "right_outer_knee": (0.4, 0.7), "left_outer_knee": (0.6, 0.7),
        "hip_joint_mid_right": (0.45, 0.56), "hip_joint_mid_left": (0.55, 0.56),
    }
    return [landmark(item.name, *overrides.get(item.name, (item.x, item.y))) for item in points]


def test_keypoint_ontology_is_complete() -> None:
    assert len(KEYPOINTS) == 105
    assert KEYPOINTS[0] == "crown"
    assert KEYPOINTS[-1] == "shoulder_joint_upper_inside_left"


def test_qc_detects_off_center_body() -> None:
    points = complete_landmarks()
    points = [landmark(item.name, item.x + 0.12 if item.name in {"crown", "groin", "lower_groin"} else item.x, item.y) for item in points]
    criteria = assess_landmarks(points, (0.02, 0.02, 0.98, 0.98))
    positioning = next(item for item in criteria if item.id == "positioning")
    assert positioning.status == "fail"
    assert summarize(criteria)[0] == "rejected"


def test_artifacts_remain_a_safe_review_gate() -> None:
    criteria = assess_landmarks(complete_landmarks(), (0.03, 0.03, 0.97, 0.97))
    artifacts = next(item for item in criteria if item.id == "artifacts")
    assert artifacts.status == "warning"
    assert summarize(criteria)[0] == "review"
