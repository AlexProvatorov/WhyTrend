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

# Normal-distribution consistency constant for the median absolute deviation.
_MAD_SCALE = 1.4826


class ZScoreDetector(BaseDetector):
    """Detect anomalies using a robust center and scale.

    The center is the median. The scale is ``1.4826 * MAD`` so one spike does
    not inflate the threshold. When the MAD is zero but the series still
    varies, the population standard deviation is the fallback.
    """

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

        located = _robust_location(values)
        if located is None:
            return self._empty_result(series, reason="zero_variance")
        center, scale = located

        detections: list[Detection] = []
        for index in range(len(values)):
            timestamp = values.index[index]
            value = float(values.iloc[index])
            z_score = abs((value - center) / scale)
            if z_score < self._threshold:
                continue

            score = zscore_to_confidence(z_score, self._threshold)
            detections.append(
                Detection(
                    timestamp=pd.Timestamp(timestamp).to_pydatetime(),
                    anomaly_type=anomaly_type_from_residual(value, center),
                    value=value,
                    score=score,
                    expected_value=center,
                    severity=score_to_severity(score),
                    index=index,
                )
            )

        detections.sort(
            key=lambda item: (item.score, abs(item.value - center)),
            reverse=True,
        )
        return DetectionResult(
            detections=detections,
            detector_name=self.name,
            series_name=series.name,
            metadata={
                "threshold": self._threshold,
                "center": center,
                "scale": scale,
                "detection_count": len(detections),
            },
        )

    def _empty_result(self, series: TimeSeries, *, reason: str) -> DetectionResult:
        return DetectionResult(
            detections=[],
            detector_name=self.name,
            series_name=series.name,
            metadata={"reason": reason},
        )


def _robust_location(values: pd.Series) -> tuple[float, float] | None:
    center = float(values.median())
    mad = float((values - center).abs().median())
    if mad > 0.0:
        return center, _MAD_SCALE * mad
    std = float(values.std(ddof=0))
    if std == 0.0:
        return None
    return center, std
