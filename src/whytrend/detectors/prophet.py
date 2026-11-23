"""Prophet-based anomaly detector."""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd

from whytrend.core.models import Detection, DetectionResult
from whytrend.core.protocols import BaseDetector
from whytrend.core.series import TimeSeries
from whytrend.detectors._utils import anomaly_type_from_residual, clamp_score, score_to_severity


class _DropMissingPlotly(logging.Filter):
    """Hide Prophet's optional-plot import noise. Real fit errors still pass."""

    def filter(self, record: logging.LogRecord) -> bool:
        return "Importing plotly failed" not in record.getMessage()


class ProphetDetector(BaseDetector):
    """Detect anomalies using Facebook Prophet forecast intervals."""

    def __init__(
        self,
        interval_width: float = 0.95,
        *,
        min_points: int = 10,
        prophet_kwargs: dict[str, Any] | None = None,
    ) -> None:
        if not 0 < interval_width < 1:
            msg = "interval_width must be between 0 and 1"
            raise ValueError(msg)
        if min_points < 2:
            msg = "min_points must be >= 2"
            raise ValueError(msg)

        self._interval_width = interval_width
        self._min_points = min_points
        self._prophet_kwargs = dict(prophet_kwargs or {})

    @property
    def name(self) -> str:
        return "prophet"

    def detect(self, series: TimeSeries) -> DetectionResult:
        if len(series) < self._min_points:
            return self._empty_result(series, reason="insufficient_points")

        frame = series.to_dataframe().rename(columns={"timestamp": "ds", "value": "y"})
        forecast = self._fit_and_predict(frame)
        merged = frame.merge(forecast, on="ds", how="left")

        detections: list[Detection] = []
        for index, row in merged.iterrows():
            actual = float(row["y"])
            expected = float(row["yhat"])
            lower = float(row["yhat_lower"])
            upper = float(row["yhat_upper"])

            if lower <= actual <= upper:
                continue

            band_width = max(upper - lower, 1e-9)
            distance = actual - upper if actual > upper else lower - actual
            score = clamp_score(distance / band_width)
            detections.append(
                Detection(
                    timestamp=row["ds"].to_pydatetime(),
                    anomaly_type=anomaly_type_from_residual(actual, expected),
                    value=actual,
                    score=score,
                    expected_value=expected,
                    severity=score_to_severity(score),
                    index=int(index) if isinstance(index, int) else None,
                )
            )

        detections.sort(key=lambda item: item.score, reverse=True)
        return DetectionResult(
            detections=detections,
            detector_name=self.name,
            series_name=series.name,
            metadata={
                "interval_width": self._interval_width,
                "model": "prophet",
            },
        )

    def _fit_and_predict(self, frame: pd.DataFrame) -> pd.DataFrame:
        _install_prophet_log_filter()
        try:
            from prophet import Prophet
        except ImportError as exc:
            msg = "ProphetDetector requires prophet; install with: pip install 'whytrend[prophet]'"
            raise ImportError(msg) from exc

        training = frame.copy()
        training["ds"] = _naive_utc(training["ds"])
        model = Prophet(interval_width=self._interval_width, **self._prophet_kwargs)
        model.fit(training)
        forecast: pd.DataFrame = model.predict(training[["ds"]])
        result = forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].copy()
        result["ds"] = _utc_timestamps(result["ds"])
        return result

    def _empty_result(self, series: TimeSeries, *, reason: str) -> DetectionResult:
        return DetectionResult(
            detections=[],
            detector_name=self.name,
            series_name=series.name,
            metadata={"reason": reason},
        )


def _install_prophet_log_filter() -> None:
    for name in ("prophet", "prophet.plot"):
        logger = logging.getLogger(name)
        if any(isinstance(item, _DropMissingPlotly) for item in logger.filters):
            continue
        logger.addFilter(_DropMissingPlotly())


def _naive_utc(values: pd.Series) -> pd.Series:
    """UTC wall time without a timezone, which is the only form Prophet accepts."""
    timestamps = pd.to_datetime(values, utc=True)
    return timestamps.dt.tz_localize(None)


def _utc_timestamps(values: pd.Series) -> pd.Series:
    timestamps = pd.to_datetime(values)
    if timestamps.dt.tz is None:
        return timestamps.dt.tz_localize("UTC")
    return timestamps.dt.tz_convert("UTC")
