from __future__ import annotations

from dataclasses import dataclass
from statistics import median

from .keypoints import Landmark


Localized = dict[str, str]


def text(ru: str, en: str) -> Localized:
    return {"ru": ru, "en": en}


@dataclass(frozen=True)
class Criterion:
    id: str
    title: Localized
    detail: Localized
    status: str
    confidence: int
    code: str | None = None

    def as_dict(self) -> dict[str, object]:
        value: dict[str, object] = {
            "id": self.id,
            "title": self.title,
            "detail": self.detail,
            "status": self.status,
            "confidence": self.confidence,
        }
        if self.code:
            value["code"] = self.code
        return value


PAIR_NAMES = (
    ("right_shoulder", "left_shoulder"),
    ("right_elbow", "left_elbow"),
    ("right_waist", "left_waist"),
    ("right_hip_skin", "left_hip_skin"),
    ("right_outer_knee", "left_outer_knee"),
    ("hip_joint_mid_right", "hip_joint_mid_left"),
)


def _status(value: float, warning: float, failure: float) -> str:
    if value >= failure:
        return "fail"
    if value >= warning:
        return "warning"
    return "pass"


def assess_landmarks(
    landmarks: list[Landmark],
    content_bbox: tuple[float, float, float, float] | None,
) -> list[Criterion]:
    by_name = {item.name: item for item in landmarks if item.visible}
    mean_confidence = sum(item.confidence for item in landmarks) / max(1, len(landmarks))
    low_confidence_fraction = sum(item.confidence < 0.5 for item in landmarks) / max(1, len(landmarks))
    landmark_status = "pass" if mean_confidence >= 0.8 and low_confidence_fraction <= 0.1 else "warning" if mean_confidence >= 0.6 else "fail"
    criteria = [
        Criterion(
            "landmark-confidence",
            text("Анатомические ориентиры", "Anatomical landmarks"),
            text(
                f"Модель определила {len(by_name)} из 105 ориентиров; средняя уверенность {mean_confidence:.1%}.",
                f"The model located {len(by_name)} of 105 landmarks with {mean_confidence:.1%} mean confidence.",
            ),
            landmark_status,
            round(mean_confidence * 100),
            "landmark_low_confidence" if landmark_status != "pass" else None,
        )
    ]

    midline = [by_name.get(name) for name in ("crown", "groin", "lower_groin")]
    midline = [item for item in midline if item is not None]
    center_deviation = abs(median(item.x for item in midline) - 0.5) if midline else 1.0
    center_status = _status(center_deviation, 0.04, 0.08)
    criteria.append(
        Criterion(
            "positioning",
            text("Центрирование", "Centering"),
            text(
                f"Отклонение средней линии от центра поля: {center_deviation:.1%} ширины изображения.",
                f"Body midline deviation from the image center is {center_deviation:.1%} of image width.",
            ),
            center_status,
            round(max(0.0, 1.0 - center_deviation * 4) * 100),
            "body_off_center" if center_status != "pass" else None,
        )
    )

    asymmetry_values: list[float] = []
    for right_name, left_name in PAIR_NAMES:
        right, left = by_name.get(right_name), by_name.get(left_name)
        if right and left:
            asymmetry_values.append(abs(right.y - left.y))
    asymmetry = median(asymmetry_values) if asymmetry_values else 1.0
    symmetry_status = _status(asymmetry, 0.025, 0.06)
    criteria.append(
        Criterion(
            "symmetry",
            text("Симметрия укладки", "Positioning symmetry"),
            text(
                f"Медианное вертикальное расхождение парных ориентиров: {asymmetry:.1%} высоты изображения.",
                f"Median vertical mismatch between paired landmarks is {asymmetry:.1%} of image height.",
            ),
            symmetry_status,
            round(max(0.0, 1.0 - asymmetry * 5) * 100),
            "body_asymmetry" if symmetry_status != "pass" else None,
        )
    )

    if content_bbox:
        left, top, right, bottom = content_bbox
        minimum_margin = min(left, top, 1.0 - right, 1.0 - bottom)
        coverage_status = "fail" if minimum_margin <= 0.002 else "warning" if minimum_margin <= 0.01 else "pass"
        coverage_confidence = 94
        coverage_detail = text(
            f"Минимальный отступ анатомии от границы кадра: {minimum_margin:.1%}.",
            f"Minimum anatomy-to-frame margin is {minimum_margin:.1%}.",
        )
    else:
        coverage_status, coverage_confidence = "warning", 50
        coverage_detail = text(
            "Граница тела не выделена надёжно; охват требует экспертной проверки.",
            "The body boundary was not extracted reliably; coverage needs expert review.",
        )
    criteria.append(
        Criterion(
            "coverage",
            text("Анатомический охват", "Anatomical coverage"),
            coverage_detail,
            coverage_status,
            coverage_confidence,
            "anatomy_cropped" if coverage_status != "pass" else None,
        )
    )

    criteria.append(
        Criterion(
            "artifacts",
            text("Артефакты", "Artifacts"),
            text(
                "Upstream-модель не обучена классифицировать металл, движение и внешние предметы; требуется отдельная проверка.",
                "The upstream model was not trained to classify metal, motion, or external objects; a separate review is required.",
            ),
            "warning",
            100,
            "artifact_model_unavailable",
        )
    )
    return criteria


def summarize(criteria: list[Criterion]) -> tuple[str, int, int]:
    status = "rejected" if any(item.status == "fail" for item in criteria) else "review" if any(item.status == "warning" for item in criteria) else "passed"
    weights = {"pass": 100, "warning": 65, "fail": 20}
    score = round(sum(weights[item.status] for item in criteria) / max(1, len(criteria)))
    confidence = round(sum(item.confidence for item in criteria) / max(1, len(criteria)))
    return status, score, confidence
