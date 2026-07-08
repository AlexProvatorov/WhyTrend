"""WhyTrend — Explainable Time Series Analysis Framework."""

from whytrend.detectors import ProphetDetector, ZScoreDetector
from whytrend.pipeline import Pipeline
from whytrend.sources import CSVSource, GoogleTrends, PandasSource, ParquetSource

__version__ = "0.1.0"
__author__ = "Alexander Provatorov"

__all__ = [
    "CSVSource",
    "GoogleTrends",
    "PandasSource",
    "ParquetSource",
    "Pipeline",
    "ProphetDetector",
    "ZScoreDetector",
    "__version__",
]
