import pytest
from pydantic import ValidationError

from whytrend.core import (
    AnomalyType,
    Cause,
    Detection,
    DetectionResult,
    Event,
    Evidence,
    Explanation,
    Report,
    TimeSeriesPoint,
)


def test_detection_result_primary_returns_highest_score(utc_ts) -> None:
    result = DetectionResult(
        detector_name="zscore",
        detections=[
            Detection(
                timestamp=utc_ts(),
                anomaly_type=AnomalyType.SPIKE,
                value=610.0,
                score=0.7,
            ),
            Detection(
                timestamp=utc_ts(1),
                anomaly_type=AnomalyType.SPIKE,
                value=640.0,
                score=0.95,
            ),
        ],
    )

    assert result.primary is not None
    assert result.primary.score == 0.95


def test_event_rejects_invalid_window(utc_ts) -> None:
    with pytest.raises(ValidationError):
        Event(
            anomaly_type=AnomalyType.SPIKE,
            timestamp=utc_ts(),
            window_start=utc_ts(24),
            window_end=utc_ts(),
            keyword="Python",
            value=610.0,
            detection_score=0.9,
        )


def test_report_json_round_trip() -> None:
    explanation = Explanation(
        event_id="evt-1",
        anomaly_type=AnomalyType.SPIKE,
        confidence=0.94,
        summary="Interest in 'Python' spiked due to Python 3.13 release.",
        causes=[
            Cause(
                source="Google News",
                score=0.95,
                url="https://example.com/python-3-13",
                title="Python 3.13 released",
            )
        ],
    )
    report = Report(
        executive_summary=explanation.summary,
        explanations=[explanation],
        series_name="python_interest",
        keyword="Python",
    )

    restored = Report.model_validate_json(report.to_json())

    assert restored.executive_summary == explanation.summary
    assert restored.explanations[0].causes[0].url.endswith("python-3-13")


def test_evidence_score_must_be_between_zero_and_one() -> None:
    with pytest.raises(ValidationError):
        Evidence(
            title="Release notes",
            url="https://example.com",
            source_name="GitHub",
            relevance_score=1.5,
        )


def test_time_series_point_accepts_timezone_aware_timestamp(utc_ts) -> None:
    point = TimeSeriesPoint(timestamp=utc_ts(), value=120.0)

    assert point.timestamp.tzinfo is not None
