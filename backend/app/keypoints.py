from __future__ import annotations

from dataclasses import dataclass

from third_party.dxa_pointplacement.dxa_points import dataset_info


KEYPOINTS: tuple[str, ...] = tuple(
    dataset_info["keypoint_info"][index]["name"]
    for index in sorted(dataset_info["keypoint_info"])
)

if len(KEYPOINTS) != 105:
    raise RuntimeError(f"Expected 105 DXA keypoints, found {len(KEYPOINTS)}")


@dataclass(frozen=True)
class Landmark:
    name: str
    x: float
    y: float
    confidence: float
    visible: bool

    def as_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "x": round(self.x, 6),
            "y": round(self.y, 6),
            "confidence": round(self.confidence, 4),
            "visible": self.visible,
        }


def normalize_landmarks(points, scores, width: int, height: int) -> list[Landmark]:
    if len(points) != len(KEYPOINTS):
        raise ValueError(f"Model returned {len(points)} points; expected {len(KEYPOINTS)}")
    result: list[Landmark] = []
    for index, name in enumerate(KEYPOINTS):
        x, y = points[index]
        confidence = float(scores[index]) if scores is not None else 0.0
        result.append(
            Landmark(
                name=name,
                x=max(0.0, min(1.0, float(x) / max(1, width))),
                y=max(0.0, min(1.0, float(y) / max(1, height))),
                confidence=max(0.0, min(1.0, confidence)),
                visible=confidence >= 0.25,
            )
        )
    return result
