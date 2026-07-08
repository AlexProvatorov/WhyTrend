"""Load time series data from an in-memory pandas DataFrame."""

from __future__ import annotations

from typing import Any

import pandas as pd

from whytrend.core.protocols import BaseSource
from whytrend.core.series import TimeSeries
from whytrend.sources._io import series_from_dataframe


class PandasSource(BaseSource):
    """Build a time series from an existing pandas DataFrame."""

    def __init__(
        self,
        frame: pd.DataFrame,
        *,
        time_col: str = "timestamp",
        value_col: str = "value",
        name: str = "",
        keyword: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self._frame = frame
        self._time_col = time_col
        self._value_col = value_col
        self._name = name
        self._keyword = keyword or name
        self._metadata = dict(metadata or {})

    @property
    def name(self) -> str:
        return "pandas"

    async def load(self) -> TimeSeries:
        return series_from_dataframe(
            self._frame.copy(),
            time_col=self._time_col,
            value_col=self._value_col,
            name=self._name,
            keyword=self._keyword,
            source_name=self.name,
            metadata=self._metadata,
        )
