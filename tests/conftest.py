from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Callable

import pandas as pd
import pytest

from whytrend.core import TimeSeries

UTC = timezone.utc
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
