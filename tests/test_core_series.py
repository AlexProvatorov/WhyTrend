"""Tests for the pandas-backed TimeSeries container."""

from datetime import datetime, timezone

import pandas as pd
import pytest

from whytrend.core import TimeSeries, TimeSeriesPoint


def _make_frame() -> pd.DataFrame:
    timestamps = pd.date_range("2026-01-01", periods=5, freq="D", tz="UTC")
    return pd.DataFrame({"timestamp": timestamps, "value": [120.0, 123.0, 121.0, 610.0, 640.0]})


def test_from_dataframe_builds_datetime_indexed_series() -> None:
    series = TimeSeries.from_dataframe(
        _make_frame(),
        time_col="timestamp",
        value_col="value",
        name="python_interest",
        keyword="Python",
    )

    assert series.name == "python_interest"
    assert series.keyword == "Python"
    assert len(series) == 5
    assert series.data.index.tz is not None
    assert series.data.iloc[-1] == 640.0


def test_to_schema_round_trip_preserves_points() -> None:
    original = TimeSeries.from_dataframe(
        _make_frame(),
        time_col="timestamp",
        value_col="value",
        name="python_interest",
        keyword="Python",
        metadata={"source": "csv"},
    )

    restored = TimeSeries.from_schema(original.to_schema())

    assert restored.name == original.name
    assert restored.keyword == original.keyword
    assert restored.metadata == original.metadata
    assert list(restored.data) == list(original.data)


def test_window_extracts_inclusive_slice() -> None:
    series = TimeSeries.from_dataframe(_make_frame(), time_col="timestamp", value_col="value")

    start = datetime(2026, 1, 3, tzinfo=timezone.utc)
    end = datetime(2026, 1, 4, tzinfo=timezone.utc)
    sliced = series.window(start, end)

    assert len(sliced) == 2
    assert sliced.data.iloc[0] == 121.0
    assert sliced.data.iloc[1] == 610.0


def test_from_points_rejects_empty_input() -> None:
    with pytest.raises(ValueError, match="points cannot be empty"):
        TimeSeries.from_points([])


def test_from_points_accepts_time_series_point_models() -> None:
    points = [
        TimeSeriesPoint(timestamp=datetime(2026, 1, 1, tzinfo=timezone.utc), value=120.0),
        TimeSeriesPoint(timestamp=datetime(2026, 1, 2, tzinfo=timezone.utc), value=123.0),
    ]

    series = TimeSeries.from_points(points, name="demo")

    assert len(series) == 2
    assert series.name == "demo"
