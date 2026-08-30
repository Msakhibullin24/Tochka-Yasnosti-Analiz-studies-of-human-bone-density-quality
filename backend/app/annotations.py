from __future__ import annotations

from datetime import datetime, timezone
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


Protocol = Literal["spine_pa", "hip_left", "hip_right", "total_body", "unsupported"]
Action = Literal["accept", "review", "repeat"]
Severity = Literal["none", "minor", "major", "critical"]
Confidence = Literal["low", "medium", "high"]
GeometryType = Literal["polygon", "polyline", "box"]
DefectCode = Literal[
    "wrong_protocol", "spine_off_center", "spine_tilt", "spine_rotation",
    "spine_incomplete_coverage", "vertebra_numbering", "vertebra_roi",
    "vertebra_exclusion", "hip_rotation", "hip_abduction_adduction",
    "femur_axis", "hip_incomplete_coverage", "femoral_neck_roi",
    "total_hip_roi", "edge_detection", "soft_tissue_boundary",
    "hyperdense_artifact", "hypodense_artifact", "motion_artifact",
    "external_object", "body_off_center", "body_asymmetry",
    "total_body_incomplete_coverage", "landmark_low_confidence",
    "artifact_model_unavailable",
]


class DefectAnnotation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    code: DefectCode
    present: bool
    severity: Severity
    action: Action | None = None
    confidence: float | None = Field(default=None, ge=0, le=1)
    comment: str = Field(default="", max_length=1000)

    @model_validator(mode="after")
    def severity_matches_presence(self) -> "DefectAnnotation":
        if not self.present and self.severity != "none":
            raise ValueError("Absent defects must use severity 'none'")
        if self.present and self.severity == "none":
            raise ValueError("Present defects must have a non-zero severity")
        return self


class LandmarkAnnotation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1, max_length=80)
    x: float = Field(ge=0, le=1)
    y: float = Field(ge=0, le=1)
    visible: bool = True


class RegionAnnotation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1, max_length=80)
    geometryType: GeometryType
    points: list[tuple[float, float]] = Field(min_length=2, max_length=128)

    @model_validator(mode="after")
    def validate_geometry(self) -> "RegionAnnotation":
        if any(not 0 <= coordinate <= 1 for point in self.points for coordinate in point):
            raise ValueError("Region coordinates must be normalized to [0, 1]")
        if self.geometryType == "box" and len(self.points) != 2:
            raise ValueError("Box regions require exactly two corner points")
        if self.geometryType == "polygon" and len(self.points) < 3:
            raise ValueError("Polygon regions require at least three points")
        return self


class ExpertAnnotation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    readerId: str = Field(min_length=1, max_length=120)
    readIndex: int = Field(default=1, ge=1, le=99)
    confidence: Confidence
    createdAt: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    adjudicated: bool = False
    comment: str = Field(default="", max_length=2000)


class AnnotationDocument(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schemaVersion: Literal["1.0.0"] = "1.0.0"
    studyId: str = Field(pattern=r"^ST-[A-F0-9]{20}$")
    patientGroupId: str = Field(pattern=r"^PG-[A-F0-9]{20}$")
    deviceGroup: str | None = Field(default=None, pattern=r"^DV-[A-F0-9]{20}$")
    siteGroup: str | None = None
    protocol: Protocol
    evaluable: bool = True
    notEvaluableReason: str = Field(default="", max_length=1000)
    overallAction: Action
    defects: list[DefectAnnotation] = Field(default_factory=list)
    landmarks: list[LandmarkAnnotation] = Field(default_factory=list)
    regions: list[RegionAnnotation] = Field(default_factory=list)
    expert: ExpertAnnotation

    @model_validator(mode="after")
    def evaluability_has_reason(self) -> "AnnotationDocument":
        if not self.evaluable and not self.notEvaluableReason.strip():
            raise ValueError("A non-evaluable study requires notEvaluableReason")
        return self
