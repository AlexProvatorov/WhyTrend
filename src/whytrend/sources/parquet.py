"""Load time series data from Parquet files."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from whytrend.core.protocols import BaseSource
from whytrend.core.series import TimeSeries
from whytrend.sources._io import read_parquet, series_from_dataframe


class ParquetSource(BaseSource):
    """Read a time series from a Parquet file."""

    def __init__(
        self,
        path: str | Path,
        *,
        time_col: str = "timestamp",
        value_col: str = "value",
        name: str | None = None,
        keyword: str | None = None,
        read_parquet_kwargs: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self._path = Path(path)
        self._time_col = time_col
        self._value_col = value_col
        self._name = name or self._path.stem
        self._keyword = keyword or self._name
        self._read_parquet_kwargs = dict(read_parquet_kwargs or {})
        self._metadata = dict(metadata or {})

    @property
    def name(self) -> str:
        return "parquet"

    @property
    def path(self) -> Path:
        return self._path

    async def load(self) -> TimeSeries:
        if not self._path.is_file():
            msg = f"Parquet file not found: {self._path}"
            raise FileNotFoundError(msg)

        frame = await read_parquet(self._path, **self._read_parquet_kwargs)
        return series_from_dataframe(
            frame,
            time_col=self._time_col,
            value_col=self._value_col,
            name=self._name,
            keyword=self._keyword,
            source_name=self.name,
            metadata={**self._metadata, "path": str(self._path)},
        )
