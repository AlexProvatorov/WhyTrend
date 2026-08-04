from datetime import UTC, datetime

import pytest

from whytrend.core import AnomalyType, Detection, DetectionResult
from whytrend.events import EventBuilder


def test_build_one_clamps_window_to_series_bounds(python_interest_series, spike_detection):
    builder = EventBuilder(window_days=10)
    event = builder.build_one(spike_detection, python_interest_series)

    assert event.keyword == "Python"
    assert event.series_name == "python_interest"
    assert event.window_start == datetime(2026, 1, 1, tzinfo=UTC)
    assert event.window_end == datetime(2026, 1, 5, tzinfo=UTC)
    assert event.anomaly_type is AnomalyType.SPIKE
    assert event.detection_score == 0.95


def test_build_returns_one_event_per_detection(python_interest_series) -> None:
    result = DetectionResult(
        detector_name="stub",
        series_name="python_interest",
        detections=[
            Detection(
                timestamp=datetime(2026, 1, 4, tzinfo=UTC),
                anomaly_type=AnomalyType.SPIKE,
                value=610.0,
                score=0.8,
            ),
            Detection(
                timestamp=datetime(2026, 1, 5, tzinfo=UTC),
                anomaly_type=AnomalyType.DROP,
                value=90.0,
                score=0.6,
            ),
        ],
    )

    events = EventBuilder(window_days=1).build(result, python_interest_series)

    assert len(events) == 2
    assert events[0].anomaly_type is AnomalyType.SPIKE
    assert events[1].anomaly_type is AnomalyType.DROP


def test_builder_rejects_negative_window() -> None:
    with pytest.raises(ValueError, match="window_days must be >= 0"):
        EventBuilder(window_days=-1)
