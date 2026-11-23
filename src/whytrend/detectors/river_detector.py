"""River-based online anomaly detector."""

from __future__ import annotations

import math
from datetime import UTC, datetime
from typing import Any, Literal, Protocol

import pandas as pd

from whytrend.core.models import Detection, DetectionResult
from whytrend.core.protocols import BaseDetector
from whytrend.core.series import TimeSeries
from whytrend.detectors._utils import anomaly_type_from_residual, clamp_score, score_to_severity

ModelPreset = Literal["gaussian", "half_space_trees"]
# River's raw Gaussian score hits 0.95 at about two standard deviations.
# Map three standard deviations to the default threshold instead.
_GAUSSIAN_REFERENCE_Z = 3.0
_GAUSSIAN_REFERENCE_THRESHOLD = 0.95


class _RiverAnomalyModel(Protocol):
    def learn_one(self, *args: Any, **kwargs: Any) -> Any: ...

    def score_one(self, *args: Any, **kwargs: Any) -> float: ...


class RiverDetector(BaseDetector):
    """Detect anomalies with online River models.

    Batch mode walks a :class:`~whytrend.core.series.TimeSeries` point by point
    (``score_one`` then ``learn_one``). Streaming mode keeps model state on the
    instance via :meth:`update`.

    Each call to :meth:`detect` builds a **fresh** model so pipeline runs stay
    independent. :meth:`update` reuses the instance model until :meth:`reset`.
    Gaussian scores use a robust scale of the points already seen. The default
    threshold ``0.95`` means three of those deviations, not the model's two-sigma CDF.

    Requires the optional extra: ``pip install 'whytrend[river]'``.
    """

    def __init__(
        self,
        *,
        model: ModelPreset = "gaussian",
        threshold: float = 0.95,
        min_points: int = 10,
        seed: int | None = 42,
        river_kwargs: dict[str, Any] | None = None,
    ) -> None:
        if model not in {"gaussian", "half_space_trees"}:
            msg = "model must be one of: gaussian, half_space_trees"
            raise ValueError(msg)
        if not 0.0 < threshold <= 1.0:
            msg = "threshold must be in (0, 1]"
            raise ValueError(msg)
        if min_points < 1:
            msg = "min_points must be >= 1"
            raise ValueError(msg)

        self._model_preset = model
        self._threshold = threshold
        self._min_points = min_points
        self._seed = seed
        self._river_kwargs = dict(river_kwargs or {})

        self._stream_model: _RiverAnomalyModel | None = None
        self._stream_count = 0
        self._stream_sum = 0.0
        self._stream_history: list[float] = []

    @property
    def name(self) -> str:
        return "river"

    @property
    def threshold(self) -> float:
        return self._threshold

    def detect(self, series: TimeSeries) -> DetectionResult:
        values = series.data.astype(float)
        if len(values) < self._min_points:
            return self._empty_result(series, reason="insufficient_points")
        if float(values.std(ddof=0)) == 0.0:
            return self._empty_result(series, reason="zero_variance")

        scores = self._score_series(values)
        if self._model_preset == "gaussian":
            scores = _zero_variance_gaussian_scores(values, scores)
        detections: list[Detection] = []
        running_sum = 0.0

        for index, score in enumerate(scores):
            value = float(values.iloc[index])
            expected = (running_sum / index) if index > 0 else value
            running_sum += value

            if index < self._min_points or score < self._threshold:
                continue

            confidence = clamp_score(score)
            detections.append(
                Detection(
                    timestamp=pd.Timestamp(values.index[index]).to_pydatetime(),
                    anomaly_type=anomaly_type_from_residual(value, expected),
                    value=value,
                    score=confidence,
                    expected_value=expected,
                    severity=score_to_severity(confidence),
                    index=index,
                )
            )

        detections.sort(key=lambda item: item.score, reverse=True)
        return DetectionResult(
            detections=detections,
            detector_name=self.name,
            series_name=series.name,
            metadata={
                "model": self._model_preset,
                "threshold": self._threshold,
                "min_points": self._min_points,
                "detection_count": len(detections),
            },
        )

    def update(
        self,
        value: float,
        *,
        timestamp: datetime | None = None,
    ) -> Detection | None:
        """Score one point, then update the online model.

        Returns a :class:`~whytrend.core.models.Detection` when the anomaly score
        is at or above ``threshold`` and the warm-up (``min_points``) is done.
        """
        if self._stream_model is None:
            self._stream_model = self._create_model()
            self._stream_count = 0
            self._stream_sum = 0.0
            self._stream_history = []

        expected = self._stream_sum / self._stream_count if self._stream_count > 0 else float(value)
        observed = float(value)

        if self._stream_count < self._min_points:
            self._learn(self._stream_model, observed)
            self._remember(observed)
            return None

        score = self._score(self._stream_model, observed)
        if self._model_preset == "gaussian":
            score = _gaussian_score(observed, self._stream_history, score)
        self._learn(self._stream_model, observed)
        self._remember(observed)

        if score < self._threshold:
            return None

        confidence = clamp_score(score)
        return Detection(
            timestamp=timestamp or datetime.now(tz=UTC),
            anomaly_type=anomaly_type_from_residual(observed, expected),
            value=observed,
            score=confidence,
            expected_value=expected,
            severity=score_to_severity(confidence),
            index=self._stream_count - 1,
        )

    def reset(self) -> None:
        """Clear streaming model state created by :meth:`update`."""
        self._stream_model = None
        self._stream_count = 0
        self._stream_sum = 0.0
        self._stream_history = []

    def _score_series(self, values: pd.Series) -> list[float]:
        model = self._create_model(values)
        scores: list[float] = []
        for index in range(len(values)):
            value = float(values.iloc[index])
            if index < self._min_points:
                scores.append(0.0)
            else:
                scores.append(self._score(model, value))
            self._learn(model, value)
        return scores

    def _create_model(self, values: pd.Series | None = None) -> _RiverAnomalyModel:
        anomaly = _load_river_anomaly()
        kwargs = dict(self._river_kwargs)
        if self._model_preset == "gaussian":
            kwargs.setdefault("grace_period", self._min_points)
            return anomaly.GaussianScorer(**kwargs)  # type: ignore[no-any-return]

        if self._seed is not None and "seed" not in kwargs:
            kwargs["seed"] = self._seed
        if "limits" not in kwargs and values is not None and len(values) > 0:
            low = float(values.min())
            high = float(values.max())
            if high <= low:
                high = low + 1.0
            kwargs["limits"] = {"value": (low, high)}
        return anomaly.HalfSpaceTrees(**kwargs)  # type: ignore[no-any-return]

    def _score(self, model: _RiverAnomalyModel, value: float) -> float:
        if self._model_preset == "gaussian":
            return float(model.score_one(None, value))
        return float(model.score_one({"value": value}))

    def _learn(self, model: _RiverAnomalyModel, value: float) -> None:
        if self._model_preset == "gaussian":
            model.learn_one(None, value)
            return
        model.learn_one({"value": value})

    def _remember(self, value: float) -> None:
        self._stream_sum += value
        self._stream_history.append(value)
        self._stream_count += 1

    def _empty_result(self, series: TimeSeries, *, reason: str) -> DetectionResult:
        return DetectionResult(
            detections=[],
            detector_name=self.name,
            series_name=series.name,
            metadata={
                "reason": reason,
                "model": self._model_preset,
                "threshold": self._threshold,
                "min_points": self._min_points,
                "detection_count": 0,
            },
        )


def _zero_variance_gaussian_scores(values: pd.Series, scores: list[float]) -> list[float]:
    """Score each point against the scale of the points already seen."""
    history: list[float] = []
    adjusted: list[float] = []
    for index, raw in enumerate(values):
        value = float(raw)
        adjusted.append(_gaussian_score(value, history, scores[index]))
        history.append(value)
    return adjusted


def _gaussian_score(value: float, history: list[float], model_score: float) -> float:
    if len(history) < 2:
        return model_score
    center, scale = _location_scale(history)
    if scale <= 1e-12:
        if math.isclose(value, center, abs_tol=1e-9):
            return 0.0
        return 1.0
    z_score = abs(value - center) / scale
    return clamp_score(z_score * _GAUSSIAN_REFERENCE_THRESHOLD / _GAUSSIAN_REFERENCE_Z)


def _location_scale(history: list[float]) -> tuple[float, float]:
    """Median and the normal-consistent MAD, with std when the MAD is zero."""
    center = _median(history)
    mad = _median([abs(item - center) for item in history])
    if mad > 0.0:
        return center, 1.4826 * mad
    mean = sum(history) / len(history)
    variance = sum((item - mean) ** 2 for item in history) / len(history)
    return center, math.sqrt(variance)


def _median(values: list[float]) -> float:
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2 == 1:
        return ordered[mid]
    return (ordered[mid - 1] + ordered[mid]) / 2


def _load_river_anomaly() -> Any:
    try:
        from river import anomaly
    except ImportError as exc:
        msg = "RiverDetector requires river; install with: pip install 'whytrend[river]'"
        raise ImportError(msg) from exc
    return anomaly
