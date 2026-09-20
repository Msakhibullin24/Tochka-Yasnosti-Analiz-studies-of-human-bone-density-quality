"""Expert revisions are separate from immutable automatic results."""
from __future__ import annotations

import math
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .model import CRITERIA


class Point(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    x: float = Field(ge=0, le=1)
    y: float = Field(ge=0, le=1)


class Region(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str = Field(min_length=1, max_length=80)
    kind: Literal["point", "line", "polyline", "polygon"]
    points: list[Point] = Field(min_length=1, max_length=128)
    note: str = Field(default="", max_length=1000)
    roi_edge: Literal["left", "right"] | None = None

    @model_validator(mode="after")
    def geometry_valid(self):
        if self.roi_edge and self.kind != "polygon":
            raise ValueError("ROI margin evaluation requires a polygon")
        if self.kind == "point" and len(self.points) != 1:
            raise ValueError("a landmark requires one point")
        if self.kind == "line" and len(self.points) != 2:
            raise ValueError("a line requires two points")
        if self.kind == "polyline" and len(self.points) < 2:
            raise ValueError("a polyline requires at least two points")
        if self.kind == "polygon" and len(self.points) < 3:
            raise ValueError("a polygon requires at least three points")
        return self


class Followup(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str = Field(min_length=1, max_length=80)
    kind: Literal["markup", "reprocess", "second_opinion", "other"]
    question: str = Field(min_length=1, max_length=1000)
    state: Literal["open", "done"] = "open"
    resolution: str = Field(default="", max_length=1000)

    @model_validator(mode="after")
    def complete(self):
        if not self.question.strip() or (self.state == "done" and not self.resolution.strip()):
            raise ValueError("action requires a question and a completed action requires a resolution")
        return self


class Review(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    expected_revision: int = Field(ge=0)
    author: str = Field(min_length=1, max_length=120)
    status: Literal["draft", "confirmed", "not_evaluable"]
    quality_class: Literal[0, 1] | None = None
    violations: list[str] = Field(default_factory=list, max_length=6)
    comment: str = Field(default="", max_length=2000)
    geometry: list[Region] = Field(default_factory=list, max_length=40)
    followups: list[Followup] = Field(default_factory=list, max_length=30)

    @model_validator(mode="after")
    def consistent(self):
        if not self.author.strip():
            raise ValueError("author is required")
        if self.status == "confirmed" and self.quality_class is None:
            raise ValueError("confirmed review requires quality_class")
        if self.status == "not_evaluable" and (not self.comment.strip() or self.quality_class is not None):
            raise ValueError("not evaluable requires a reason and no quality class")
        if self.quality_class in (0, None) and self.violations:
            raise ValueError("normal review cannot contain violations")
        if self.status == "confirmed" and self.quality_class == 1 and not self.violations:
            raise ValueError("a violation requires at least one type")
        if len(set(self.violations)) != len(self.violations):
            raise ValueError("duplicate violations")
        names = [g.name for g in self.geometry]
        if len(set(names)) != len(names):
            raise ValueError("geometry names must be unique")
        ids = [item.id for item in self.followups]
        if len(set(ids)) != len(ids):
            raise ValueError("action ids must be unique")
        return self

    def for_region(self, region: str):
        allowed = set(CRITERIA["spine" if region == "spine" else "hip"])
        if region != "spine":
            allowed.add("hip_metal_implant")
        if not set(self.violations) <= allowed:
            raise ValueError("violation does not belong to this anatomical region")


def measurements(geometry: list[Region], width: int, height: int, sx: float, sy: float) -> dict:
    """Explicit expert measurements only; do not pretend to re-run anatomical inference."""
    out = {}
    for g in geometry:
        if g.kind == "line":
            a, b = g.points
            dx = (b.x - a.x) * (width - 1) * sx
            dy = (b.y - a.y) * (height - 1) * sy
            out[g.name] = {"length_mm": round(math.hypot(dx, dy), 3),
                           "angle_from_vertical_deg": round(math.degrees(math.atan2(dx, dy)), 3)}
    return out
