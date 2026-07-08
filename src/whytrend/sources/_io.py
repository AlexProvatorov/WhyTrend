"""Shared helpers for source implementations."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import pandas as pd

from whytrend.core.series import TimeSeries


def series_from_dataframe(
    df: pd.DataFrame,
    *,
    time_col: str,
    value_col: str,
    name: str,
    keyword: str,
    source_name: str,
    metadata: dict[str, Any] | None = None,
) -> TimeSeries:
    """Convert a DataFrame into a :class:`TimeSeries` with source metadata."""
    meta = dict(metadata or {})
    meta["source"] = source_name
    return TimeSeries.from_dataframe(
        df,
        time_col=time_col,
        value_col=value_col,
        name=name,
        keyword=keyword or name,
        metadata=meta,
    )


async def read_csv(path: Path, **kwargs: Any) -> pd.DataFrame:
    return await asyncio.to_thread(lambda: pd.read_csv(path, **kwargs))


async def read_parquet(path: Path, **kwargs: Any) -> pd.DataFrame:
    return await asyncio.to_thread(lambda: pd.read_parquet(path, **kwargs))
