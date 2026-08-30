from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

from .keypoints import Landmark
from .preprocessing import PreparedStudy
from .qc import Criterion, assess_landmarks


@dataclass(frozen=True)
class AdapterPrediction:
    protocol: str
    landmarks: tuple[Landmark, ...]
    criteria: tuple[Criterion, ...]
    metadata: dict[str, object]


class ModelAdapter(Protocol):
    key: str
    protocol: str

    def preprocess(self, study: PreparedStudy) -> Any: ...

    def predict(self, prepared_input: Any) -> Any: ...

    def postprocess(self, raw_prediction: Any, study: PreparedStudy) -> AdapterPrediction: ...

    def build_overlay(self, prediction: AdapterPrediction) -> list[dict[str, object]]: ...


class TotalBodyLandmarkAdapter:
    key = "hawaii-ai/dxa-pointplacement@7ac19eb"
    protocol = "total-body"

    def __init__(self, inference_model: Any):
        self.model = inference_model

    def preprocess(self, study: PreparedStudy) -> Any:
        return study.image

    def predict(self, prepared_input: Any) -> list[Landmark]:
        return self.model.predict(prepared_input)

    def postprocess(self, raw_prediction: list[Landmark], study: PreparedStudy) -> AdapterPrediction:
        landmarks = tuple(raw_prediction)
        criteria = tuple(assess_landmarks(list(landmarks), study.content_bbox))
        return AdapterPrediction(
            protocol=self.protocol,
            landmarks=landmarks,
            criteria=criteria,
            metadata={"adapterKey": self.key, "landmarkCount": len(landmarks)},
        )

    def build_overlay(self, prediction: AdapterPrediction) -> list[dict[str, object]]:
        return [landmark.as_dict() for landmark in prediction.landmarks]

    def run(self, study: PreparedStudy) -> AdapterPrediction:
        return self.postprocess(self.predict(self.preprocess(study)), study)
