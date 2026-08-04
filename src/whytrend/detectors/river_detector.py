"""River-based online anomaly detector."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any, Literal, Protocol

import pandas as pd

from whytrend.core.models import Detection, DetectionResult
from whytrend.core.protocols import BaseDetector
from whytrend.core.series import TimeSeries
from whytrend.detectors._utils import anomaly_type_from_residual, clamp_score, score_to_severity

ModelPreset = Literal["gaussian", "half_space_trees"]


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

        scores = self._score_series(values)
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

        expected = self._stream_sum / self._stream_count if self._stream_count > 0 else float(value)

        if self._stream_count < self._min_points:
            self._learn(self._stream_model, float(value))
            self._stream_sum += float(value)
            self._stream_count += 1
            return None

        score = self._score(self._stream_model, float(value))
        self._learn(self._stream_model, float(value))
        self._stream_sum += float(value)
        self._stream_count += 1

        if score < self._threshold:
            return None

        confidence = clamp_score(score)
        return Detection(
            timestamp=timestamp or datetime.now(tz=UTC),
            anomaly_type=anomaly_type_from_residual(float(value), expected),
            value=float(value),
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

    def _score_series(self, values: pd.Series) -> list[float]:
        model = self._create_model()
        scores: list[float] = []
        for index in range(len(values)):
            value = float(values.iloc[index])
            if index < self._min_points:
                scores.append(0.0)
            else:
                scores.append(self._score(model, value))
            self._learn(model, value)
        return scores

    def _create_model(self) -> _RiverAnomalyModel:
        try:
            from river import anomaly
        except ImportError as exc:
            msg = "RiverDetector requires river; install with: pip install 'whytrend[river]'"
            raise ImportError(msg) from exc

        kwargs = dict(self._river_kwargs)
        if self._model_preset == "gaussian":
            return anomaly.GaussianScorer(**kwargs)  # type: ignore[no-any-return]

        if self._seed is not None and "seed" not in kwargs:
            kwargs["seed"] = self._seed
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
