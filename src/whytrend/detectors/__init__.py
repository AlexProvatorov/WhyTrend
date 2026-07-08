"""Anomaly and change-point detectors."""

from whytrend.detectors.prophet import ProphetDetector
from whytrend.detectors.zscore import ZScoreDetector

__all__ = [
    "ProphetDetector",
    "ZScoreDetector",
]
