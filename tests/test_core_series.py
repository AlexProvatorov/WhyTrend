from datetime import UTC, datetime

import pytest

from whytrend.core import TimeSeries, TimeSeriesPoint


def test_from_dataframe_builds_datetime_indexed_series(spike_dataframe) -> None:
    series = TimeSeries.from_dataframe(
        spike_dataframe,
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


def test_to_schema_round_trip_preserves_points(python_interest_series) -> None:
    restored = TimeSeries.from_schema(python_interest_series.to_schema())

    assert restored.name == python_interest_series.name
    assert restored.keyword == python_interest_series.keyword
    assert restored.metadata == python_interest_series.metadata
    assert list(restored.data) == list(python_interest_series.data)


def test_window_extracts_inclusive_slice(python_interest_series, spike_window) -> None:
    start, end = spike_window
    sliced = python_interest_series.window(start, end)

    assert len(sliced) == 2
    assert sliced.data.iloc[0] == 121.0
    assert sliced.data.iloc[1] == 610.0


def test_from_points_rejects_empty_input() -> None:
    with pytest.raises(ValueError, match="points cannot be empty"):
        TimeSeries.from_points([])


def test_from_points_accepts_time_series_point_models() -> None:
    points = [
        TimeSeriesPoint(timestamp=datetime(2026, 1, 1, tzinfo=UTC), value=120.0),
        TimeSeriesPoint(timestamp=datetime(2026, 1, 2, tzinfo=UTC), value=123.0),
    ]

    series = TimeSeries.from_points(points, name="demo")

    assert len(series) == 2
    assert series.name == "demo"
