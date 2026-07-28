from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime, timedelta

import pandas as pd
import pytest

from tests.stubs import StubCollector, StubDetector, StubExplainer, StubSource
from whytrend.core import AnomalyType, Detection, Evidence, TimeSeries
from whytrend.pipeline import Pipeline

PYTHON_KEYWORD = "Python"
PYTHON_SERIES_NAME = "python_interest"
SPIKE_VALUES = [120.0, 123.0, 121.0, 610.0, 640.0]


@pytest.fixture
def utc_ts() -> Callable[[int], datetime]:
    """Return a helper that builds UTC datetimes from a fixed base."""

    base = datetime(2026, 3, 15, 12, 0, tzinfo=UTC)

    def _offset(hours: int = 0) -> datetime:
        return base + timedelta(hours=hours)

    return _offset


@pytest.fixture
def spike_dataframe() -> pd.DataFrame:
    """Sample daily series with a spike on the last two days."""
    timestamps = pd.date_range("2026-01-01", periods=len(SPIKE_VALUES), freq="D", tz="UTC")
    return pd.DataFrame({"timestamp": timestamps, "value": SPIKE_VALUES})


@pytest.fixture
def python_interest_series(spike_dataframe: pd.DataFrame) -> TimeSeries:
    """Canonical TimeSeries used across core tests."""
    return TimeSeries.from_dataframe(
        spike_dataframe,
        time_col="timestamp",
        value_col="value",
        name=PYTHON_SERIES_NAME,
        keyword=PYTHON_KEYWORD,
        metadata={"source": "csv"},
    )


@pytest.fixture
def spike_window() -> tuple[datetime, datetime]:
    """Inclusive date window around the spike in :data:`spike_dataframe`."""
    start = datetime(2026, 1, 3, tzinfo=UTC)
    end = datetime(2026, 1, 4, tzinfo=UTC)
    return start, end


@pytest.fixture
def spike_detection(utc_ts) -> Detection:
    return Detection(
        timestamp=datetime(2026, 1, 4, tzinfo=UTC),
        anomaly_type=AnomalyType.SPIKE,
        value=610.0,
        score=0.95,
        expected_value=121.0,
    )


@pytest.fixture
def sample_evidence(utc_ts) -> Evidence:
    return Evidence(
        title="Python 3.13 released",
        url="https://example.com/python-3-13",
        snippet="Python 3.13 brings performance improvements.",
        source_name="Google News",
        published_at=utc_ts(),
        relevance_score=0.9,
    )


@pytest.fixture
def pipeline_components(python_interest_series, spike_detection, sample_evidence):
    source = StubSource(python_interest_series)
    detector = StubDetector([spike_detection])
    collector = StubCollector([sample_evidence])
    explainer = StubExplainer()
    return source, detector, collector, explainer


@pytest.fixture
def configured_pipeline(pipeline_components):
    source, detector, collector, explainer = pipeline_components
    return (
        Pipeline(window_days=3)
        .add_source(source)
        .add_detector(detector)
        .add_collector(collector)
        .add_explainer(explainer)
    )
