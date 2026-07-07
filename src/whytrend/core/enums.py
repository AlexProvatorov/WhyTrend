"""Domain enumerations for WhyTrend."""

from enum import StrEnum


class AnomalyType(StrEnum):
    """Type of anomaly or structural change detected in a time series."""

    SPIKE = "spike"
    DROP = "drop"
    CHANGEPOINT = "changepoint"
    OUTLIER = "outlier"
    SEASONALITY_BREAK = "seasonality_break"


class DetectionSeverity(StrEnum):
    """Relative severity of a detection."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"
