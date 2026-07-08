"""Shared helpers for detector implementations."""

from __future__ import annotations

import math

from whytrend.core.enums import AnomalyType, DetectionSeverity


def clamp_score(value: float) -> float:
    return max(0.0, min(1.0, value))


def zscore_to_confidence(z_score: float, threshold: float) -> float:
    """Map an absolute z-score into detector confidence [0, 1]."""
    if threshold <= 0:
        return 1.0
    return clamp_score(z_score / (threshold * 2))


def score_to_severity(score: float) -> DetectionSeverity:
    if score >= 0.9:
        return DetectionSeverity.CRITICAL
    if score >= 0.75:
        return DetectionSeverity.HIGH
    if score >= 0.5:
        return DetectionSeverity.MEDIUM
    return DetectionSeverity.LOW


def anomaly_type_from_residual(value: float, expected: float) -> AnomalyType:
    if math.isclose(value, expected):
        return AnomalyType.OUTLIER
    return AnomalyType.SPIKE if value > expected else AnomalyType.DROP
