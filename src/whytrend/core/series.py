"""TimeSeries container backed by pandas."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Self

import pandas as pd

from whytrend.core.models import TimeSeriesPoint, TimeSeriesSchema


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
        return (
            self.data.rename("value")
            .reset_index(names="timestamp")
            .loc[:, ["timestamp", "value"]]
        )

    def to_schema(self) -> TimeSeriesSchema:
        """Serialize to a JSON-friendly pydantic schema."""
        points = [
            TimeSeriesPoint(timestamp=timestamp.to_pydatetime(), value=float(value))
            for timestamp, value in self.data.items()
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
        if start_ts.tzinfo is None:
            start_ts = start_ts.tz_localize("UTC")
        else:
            start_ts = start_ts.tz_convert("UTC")
        if end_ts.tzinfo is None:
            end_ts = end_ts.tz_localize("UTC")
        else:
            end_ts = end_ts.tz_convert("UTC")

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
        return self.data.index[0].to_pydatetime()

    @property
    def end(self) -> datetime:
        return self.data.index[-1].to_pydatetime()

    def __len__(self) -> int:
        return len(self.data)
