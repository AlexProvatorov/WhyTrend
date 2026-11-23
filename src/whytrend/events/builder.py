"""Build contextualizable events from detector output."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from whytrend.core.models import Detection, DetectionResult, Event
from whytrend.core.series import TimeSeries


def _detection_order(detection: Detection) -> tuple[int, datetime]:
    index = detection.index if detection.index is not None else 10**12
    return (index, _ensure_utc(detection.timestamp))


def _continues_cluster(previous: Detection, current: Detection) -> bool:
    if previous.anomaly_type != current.anomaly_type:
        return False
    if previous.index is None or current.index is None:
        return False
    return current.index == previous.index + 1


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
        """Build events, merging adjacent detections of the same type."""
        selected = list(detections if detections is not None else result.detections)
        ordered = sorted(selected, key=_detection_order)
        groups: list[list[Detection]] = []
        for detection in ordered:
            if groups and _continues_cluster(groups[-1][-1], detection):
                groups[-1].append(detection)
            else:
                groups.append([detection])
        return [self._build_group(group, series) for group in groups]

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

    def _build_group(self, group: list[Detection], series: TimeSeries) -> Event:
        peak = max(
            group,
            key=lambda item: (item.score, abs(item.value - (item.expected_value or item.value))),
        )
        event = self.build_one(peak, series)
        if len(group) == 1:
            return event

        start = min(_ensure_utc(item.timestamp) for item in group)
        end = max(_ensure_utc(item.timestamp) for item in group)
        window = timedelta(days=self.window_days)
        series_start = _ensure_utc(series.start)
        series_end = _ensure_utc(series.end)
        window_start = max(start - window, series_start)
        window_end = min(end + window, series_end)
        if window_end < window_start:
            window_start = series_start
            window_end = series_end
        return event.model_copy(
            update={
                "window_start": window_start,
                "window_end": window_end,
                "metadata": {**event.metadata, "merged_detections": len(group)},
            }
        )
