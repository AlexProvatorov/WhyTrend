"""Build contextualizable events from detector output."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from whytrend.core.models import Detection, DetectionResult, Event
from whytrend.core.series import TimeSeries


def _ensure_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


class EventBuilder:
    """Convert detections into :class:`Event` objects for context collection."""

    def __init__(self, window_days: int = 3) -> None:
        if window_days < 0:
            msg = "window_days must be >= 0"
            raise ValueError(msg)
        self.window_days = window_days

    def build(
        self,
        result: DetectionResult,
        series: TimeSeries,
        *,
        detections: list[Detection] | None = None,
    ) -> list[Event]:
        """Build one event per selected detection."""
        selected = detections if detections is not None else result.detections
        return [self.build_one(detection, series) for detection in selected]

    def build_one(self, detection: Detection, series: TimeSeries) -> Event:
        """Build a single event from one detection."""
        timestamp = _ensure_utc(detection.timestamp)
        series_start = _ensure_utc(series.start)
        series_end = _ensure_utc(series.end)
        window = timedelta(days=self.window_days)

        window_start = max(timestamp - window, series_start)
        window_end = min(timestamp + window, series_end)
        if window_end < window_start:
            window_start = series_start
            window_end = series_end

        return Event(
            anomaly_type=detection.anomaly_type,
            timestamp=timestamp,
            window_start=window_start,
            window_end=window_end,
            keyword=series.keyword or series.name,
            series_name=series.name,
            value=detection.value,
            detection_score=detection.score,
            metadata={
                "detector_expected_value": detection.expected_value,
                "detection_severity": detection.severity.value,
            },
        )
