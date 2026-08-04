"""TimeSeries container backed by pandas."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Self, cast

import pandas as pd

from whytrend.core.models import TimeSeriesPoint, TimeSeriesSchema


def _to_datetime(value: datetime | pd.Timestamp) -> datetime:
    """Convert a pandas timestamp-like value to a Python datetime."""
    if isinstance(value, datetime):
        return value
    return cast(datetime, value.to_pydatetime())


@dataclass(slots=True)
class TimeSeries:
    """In-memory time series used across the WhyTrend pipeline.

    Wraps a pandas ``Series`` with a ``DatetimeIndex`` and optional metadata
    (name, keyword for context collectors, arbitrary tags).
    """

    data: pd.Series
    name: str = ""
    keyword: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.data.index, pd.DatetimeIndex):
            msg = "TimeSeries.data must use a DatetimeIndex"
            raise TypeError(msg)
        if self.data.empty:
            msg = "TimeSeries.data cannot be empty"
            raise ValueError(msg)

    @classmethod
    def from_dataframe(
        cls,
        df: pd.DataFrame,
        *,
        time_col: str,
        value_col: str,
        name: str = "",
        keyword: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> Self:
        """Build a :class:`TimeSeries` from a pandas DataFrame."""
        if time_col not in df.columns:
            msg = f"time column '{time_col}' not found in DataFrame"
            raise KeyError(msg)
        if value_col not in df.columns:
            msg = f"value column '{value_col}' not found in DataFrame"
            raise KeyError(msg)

        frame = df[[time_col, value_col]].copy()
        frame[time_col] = pd.to_datetime(frame[time_col], utc=True)
        frame = frame.sort_values(time_col).drop_duplicates(subset=[time_col], keep="last")
        series = frame.set_index(time_col)[value_col].astype(float)
        series.index = pd.DatetimeIndex(series.index)

        return cls(
            data=series,
            name=name,
            keyword=keyword or name,
            metadata=dict(metadata or {}),
        )

    @classmethod
    def from_points(
        cls,
        points: list[TimeSeriesPoint],
        *,
        name: str = "",
        keyword: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> Self:
        """Build a :class:`TimeSeries` from pydantic point models."""
        if not points:
            msg = "points cannot be empty"
            raise ValueError(msg)

        index = pd.DatetimeIndex([point.timestamp for point in points])
        values = pd.Series([point.value for point in points], index=index, dtype=float)
        values = values[~values.index.duplicated(keep="last")].sort_index()

        return cls(
            data=values,
            name=name,
            keyword=keyword or name,
            metadata=dict(metadata or {}),
        )

    @classmethod
    def from_schema(cls, schema: TimeSeriesSchema) -> Self:
        """Reconstruct a :class:`TimeSeries` from a serialized schema."""
        return cls.from_points(
            schema.points,
            name=schema.name,
            keyword=schema.keyword,
            metadata=schema.metadata,
        )

    def to_dataframe(self) -> pd.DataFrame:
        """Return a DataFrame with ``timestamp`` and ``value`` columns."""
        frame = self.data.to_frame(name="value")
        frame.index.name = "timestamp"
        return frame.reset_index()

    def to_schema(self) -> TimeSeriesSchema:
        """Serialize to a JSON-friendly pydantic schema."""
        points = [
            TimeSeriesPoint(
                timestamp=_to_datetime(cast(pd.Timestamp, self.data.index[index])),
                value=float(self.data.iloc[index]),
            )
            for index in range(len(self.data))
        ]
        return TimeSeriesSchema(
            name=self.name,
            keyword=self.keyword,
            points=points,
            metadata=self.metadata,
        )

    def window(self, start: datetime, end: datetime) -> TimeSeries:
        """Return a slice of the series between ``start`` and ``end`` (inclusive)."""
        start_ts = pd.Timestamp(start)
        end_ts = pd.Timestamp(end)
        start_ts = (
            start_ts.tz_localize("UTC") if start_ts.tzinfo is None else start_ts.tz_convert("UTC")
        )
        end_ts = end_ts.tz_localize("UTC") if end_ts.tzinfo is None else end_ts.tz_convert("UTC")

        sliced = self.data.loc[start_ts:end_ts]

        if sliced.empty:
            msg = f"no data in window [{start}, {end}]"
            raise ValueError(msg)

        return TimeSeries(
            data=sliced,
            name=self.name,
            keyword=self.keyword,
            metadata=dict(self.metadata),
        )

    @property
    def start(self) -> datetime:
        return _to_datetime(cast(pd.Timestamp, self.data.index[0]))

    @property
    def end(self) -> datetime:
        return _to_datetime(cast(pd.Timestamp, self.data.index[-1]))

    def __len__(self) -> int:
        return len(self.data)
