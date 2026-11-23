"""Domain enumerations for WhyTrend."""

from enum import StrEnum


class AnomalyType(StrEnum):
    """Type of anomaly or structural change detected in a time series."""

    SPIKE = "spike"
    DROP = "drop"
    CHANGEPOINT = "changepoint"
    OUTLIER = "outlier"
    SEASONALITY_BREAK = "seasonality_break"


class EvidenceVerdict(StrEnum):
    """How strongly the remaining evidence supports a cause."""

    LIKELY_CAUSE = "likely_cause"
    CORRELATION_ONLY = "correlation_only"
    EVIDENCE_INSUFFICIENT = "evidence_insufficient"


class DetectionSeverity(StrEnum):
    """Relative severity of a detection."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"
