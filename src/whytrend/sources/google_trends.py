"""Load interest-over-time data from Google Trends."""

from __future__ import annotations

import asyncio
from typing import Any

import pandas as pd

from whytrend.core.protocols import BaseSource
from whytrend.core.series import TimeSeries
from whytrend.sources._io import series_from_dataframe


class GoogleTrends(BaseSource):
    """Fetch a Google Trends interest-over-time series for a keyword."""

    def __init__(
        self,
        keyword: str,
        *,
        geo: str = "",
        timeframe: str = "today 12-m",
        name: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        if not keyword.strip():
            msg = "keyword cannot be empty"
            raise ValueError(msg)

        self._keyword = keyword.strip()
        self._geo = geo
        self._timeframe = timeframe
        self._name = name or f"google_trends_{self._keyword.lower().replace(' ', '_')}"
        self._metadata = dict(metadata or {})

    @property
    def name(self) -> str:
        return "google_trends"

    @property
    def keyword(self) -> str:
        return self._keyword

    async def load(self) -> TimeSeries:
        frame = await asyncio.to_thread(self._fetch_interest_over_time)
        return series_from_dataframe(
            frame,
            time_col="timestamp",
            value_col="value",
            name=self._name,
            keyword=self._keyword,
            source_name=self.name,
            metadata={
                **self._metadata,
                "geo": self._geo,
                "timeframe": self._timeframe,
            },
        )

    def _fetch_interest_over_time(self) -> pd.DataFrame:
        try:
            from pytrends.request import TrendReq
        except ImportError as exc:
            msg = "GoogleTrends requires pytrends; install with: pip install 'whytrend[trends]'"
            raise ImportError(msg) from exc

        client = TrendReq()
        client.build_payload([self._keyword], timeframe=self._timeframe, geo=self._geo)
        frame = client.interest_over_time()

        if frame.empty:
            msg = f"Google Trends returned no data for keyword '{self._keyword}'"
            raise ValueError(msg)

        if "isPartial" in frame.columns:
            frame = frame.drop(columns=["isPartial"])

        value_col = self._keyword if self._keyword in frame.columns else frame.columns[0]
        normalized: pd.DataFrame = frame.reset_index(names="timestamp")[
            ["timestamp", value_col]
        ].rename(columns={value_col: "value"})
        return normalized
