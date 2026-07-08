from datetime import datetime, timezone
from unittest.mock import patch

import pandas as pd
import pytest

from whytrend.core import AnomalyType
from whytrend.detectors import ProphetDetector, ZScoreDetector
from whytrend.events import EventBuilder
from whytrend.pipeline import Pipeline
from whytrend.sources import PandasSource
from tests.stubs import StubCollector, StubExplainer


def test_zscore_detector_finds_spike(python_interest_series) -> None:
    detector = ZScoreDetector(threshold=1.0)
    result = detector.detect(python_interest_series)

    assert result.detector_name == "zscore"
    assert result.series_name == "python_interest"
    assert result.primary is not None
    assert result.primary.anomaly_type is AnomalyType.SPIKE
    assert result.primary.value == 640.0
    assert result.primary.expected_value == pytest.approx(322.8, rel=1e-2)
    assert result.primary.score > 0.0


def test_zscore_detector_returns_empty_for_short_series(python_interest_series) -> None:
    short_series = python_interest_series.window(
        datetime(2026, 1, 1, tzinfo=timezone.utc),
        datetime(2026, 1, 3, tzinfo=timezone.utc),
    )

    result = ZScoreDetector(min_points=5).detect(short_series)

    assert result.detections == []
    assert result.metadata["reason"] == "insufficient_points"


def test_zscore_detector_rejects_invalid_threshold() -> None:
    with pytest.raises(ValueError, match="threshold must be > 0"):
        ZScoreDetector(threshold=0)


def test_prophet_detector_finds_points_outside_forecast_band(python_interest_series) -> None:
    frame = python_interest_series.to_dataframe().rename(columns={"timestamp": "ds", "value": "y"})
    forecast = pd.DataFrame(
        {
            "ds": frame["ds"],
            "yhat": [122.0] * len(frame),
            "yhat_lower": [100.0] * len(frame),
            "yhat_upper": [140.0] * len(frame),
        }
    )

    detector = ProphetDetector(min_points=5)
    with patch.object(detector, "_fit_and_predict", return_value=forecast):
        result = detector.detect(python_interest_series)

    assert result.detector_name == "prophet"
    assert result.primary is not None
    assert result.primary.value in {610.0, 640.0}
    assert result.primary.anomaly_type is AnomalyType.SPIKE


def test_event_builder_uses_detector_output(python_interest_series) -> None:
    result = ZScoreDetector(threshold=1.0).detect(python_interest_series)
    events = EventBuilder(window_days=2).build(result, python_interest_series)

    assert len(events) >= 1
    assert events[0].keyword == "Python"
    assert events[0].detection_score == result.primary.score


@pytest.mark.asyncio
async def test_pipeline_with_zscore_detector_and_pandas_source(
    spike_dataframe,
    sample_evidence,
) -> None:
    pipeline = (
        Pipeline(window_days=2)
        .add_source(
            PandasSource(
                spike_dataframe,
                name="python_interest",
                keyword="Python",
            )
        )
        .add_detector(ZScoreDetector(threshold=1.0))
        .add_collector(StubCollector([sample_evidence]))
        .add_explainer(StubExplainer())
    )

    report = await pipeline.arun()

    assert report.keyword == "Python"
    assert report.executive_summary.startswith("Explained Python")
    assert len(report.explanations) >= 1
    assert report.metadata["detector"] == "zscore"
    assert report.metadata["source"] == "pandas"
