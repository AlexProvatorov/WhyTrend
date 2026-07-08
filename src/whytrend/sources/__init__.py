"""Time series data sources."""

from whytrend.sources.csv import CSVSource
from whytrend.sources.google_trends import GoogleTrends
from whytrend.sources.pandas_source import PandasSource
from whytrend.sources.parquet import ParquetSource

__all__ = [
    "CSVSource",
    "GoogleTrends",
    "PandasSource",
    "ParquetSource",
]
