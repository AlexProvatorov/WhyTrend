"""Z-score based anomaly detector."""

from __future__ import annotations

import pandas as pd

from whytrend.core.models import Detection, DetectionResult
from whytrend.core.protocols import BaseDetector
from whytrend.core.series import TimeSeries
from whytrend.detectors._utils import (
    anomaly_type_from_residual,
    score_to_severity,
    zscore_to_confidence,
)


class ZScoreDetector(BaseDetector):
    """Detect anomalies using global mean and standard deviation."""

    def __init__(
        self,
        threshold: float = 3.0,
        *,
        min_points: int = 5,
    ) -> None:
        if threshold <= 0:
            msg = "threshold must be > 0"
            raise ValueError(msg)
        if min_points < 2:
            msg = "min_points must be >= 2"
            raise ValueError(msg)

        self._threshold = threshold
        self._min_points = min_points

    @property
    def name(self) -> str:
        return "zscore"

    @property
    def threshold(self) -> float:
        return self._threshold

    def detect(self, series: TimeSeries) -> DetectionResult:
        values = series.data.astype(float)
        if len(values) < self._min_points:
            return self._empty_result(series, reason="insufficient_points")

        mean = float(values.mean())
        std = float(values.std(ddof=0))
        if std == 0.0:
            return self._empty_result(series, reason="zero_variance")

        detections: list[Detection] = []
        for index in range(len(values)):
            timestamp = values.index[index]
            value = float(values.iloc[index])
            z_score = abs((value - mean) / std)
            if z_score < self._threshold:
                continue

            score = zscore_to_confidence(z_score, self._threshold)
            detections.append(
                Detection(
                    timestamp=pd.Timestamp(timestamp).to_pydatetime(),
                    anomaly_type=anomaly_type_from_residual(value, mean),
                    value=value,
                    score=score,
                    expected_value=mean,
                    severity=score_to_severity(score),
                    index=index,
                )
            )

        detections.sort(key=lambda item: item.score, reverse=True)
        return DetectionResult(
            detections=detections,
            detector_name=self.name,
            series_name=series.name,
            metadata={
                "threshold": self._threshold,
                "mean": mean,
                "std": std,
            },
        )

    def _empty_result(self, series: TimeSeries, *, reason: str) -> DetectionResult:
        return DetectionResult(
            detections=[],
            detector_name=self.name,
            series_name=series.name,
            metadata={"reason": reason},
        )
