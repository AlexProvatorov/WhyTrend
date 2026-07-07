"""Pydantic domain models for detection, events, evidence, and reports."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, field_validator

from whytrend.core.enums import AnomalyType, DetectionSeverity


def _utc_now() -> datetime:
    return datetime.now(tz=timezone.utc)


class WhyTrendModel(BaseModel):
    """Base model with strict validation and JSON-friendly defaults."""

    model_config = ConfigDict(
        strict=False,
        validate_assignment=True,
        extra="forbid",
        populate_by_name=True,
    )


class TimeSeriesPoint(WhyTrendModel):
    """Single observation in a time series."""

    timestamp: datetime
    value: float


class TimeSeriesSchema(WhyTrendModel):
    """JSON-serializable representation of a :class:`~whytrend.core.series.TimeSeries`."""

    name: str = ""
    keyword: str = ""
    points: list[TimeSeriesPoint]
    metadata: dict[str, Any] = Field(default_factory=dict)


class Detection(WhyTrendModel):
    """A single anomaly or change-point detection."""

    timestamp: datetime
    anomaly_type: AnomalyType
    value: float
    score: float = Field(ge=0.0, le=1.0, description="Detector confidence in [0, 1].")
    expected_value: float | None = None
    severity: DetectionSeverity = DetectionSeverity.MEDIUM
    index: int | None = Field(default=None, ge=0)


class DetectionResult(WhyTrendModel):
    """Output of a detector run."""

    detections: list[Detection]
    detector_name: str
    series_name: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)

    @property
    def primary(self) -> Detection | None:
        """Highest-scoring detection, if any."""
        if not self.detections:
            return None
        return max(self.detections, key=lambda item: item.score)


class Event(WhyTrendModel):
    """Contextualizable event derived from one or more detections."""

    id: str = Field(default_factory=lambda: str(uuid4()))
    anomaly_type: AnomalyType
    timestamp: datetime
    window_start: datetime
    window_end: datetime
    keyword: str
    series_name: str = ""
    value: float
    detection_score: float = Field(ge=0.0, le=1.0)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("window_end")
    @classmethod
    def validate_window(cls, window_end: datetime, info) -> datetime:
        window_start = info.data.get("window_start")
        if window_start is not None and window_end < window_start:
            msg = "window_end must be >= window_start"
            raise ValueError(msg)
        return window_end


class Evidence(WhyTrendModel):
    """External context item collected for an event."""

    title: str
    url: str
    snippet: str = ""
    source_name: str
    published_at: datetime | None = None
    relevance_score: float = Field(default=0.0, ge=0.0, le=1.0)
    metadata: dict[str, Any] = Field(default_factory=dict)


class Cause(WhyTrendModel):
    """Ranked causal hypothesis supported by evidence."""

    source: str
    score: float = Field(ge=0.0, le=1.0)
    url: str
    title: str = ""
    summary: str = ""


class Explanation(WhyTrendModel):
    """Structured explanation for a single event."""

    event_id: str
    anomaly_type: AnomalyType
    confidence: float = Field(ge=0.0, le=1.0)
    summary: str
    causes: list[Cause] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class Report(WhyTrendModel):
    """Top-level pipeline output."""

    executive_summary: str
    explanations: list[Explanation] = Field(default_factory=list)
    series_name: str = ""
    keyword: str = ""
    generated_at: datetime = Field(default_factory=_utc_now)
    metadata: dict[str, Any] = Field(default_factory=dict)

    def to_json(self) -> str:
        """Return a JSON string suitable for storage or APIs."""
        return self.model_dump_json(indent=2)
