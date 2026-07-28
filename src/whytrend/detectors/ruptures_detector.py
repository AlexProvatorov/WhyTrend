"""Ruptures-based change-point detector."""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
import pandas as pd

from whytrend.core.enums import AnomalyType
from whytrend.core.models import Detection, DetectionResult
from whytrend.core.protocols import BaseDetector
from whytrend.core.series import TimeSeries
from whytrend.detectors._utils import clamp_score, score_to_severity

AlgorithmName = Literal["pelt", "binseg", "bottomup"]
CostModel = Literal["l2", "rbf", "normal", "l1", "ar"]


class RupturesDetector(BaseDetector):
    """Detect structural breaks with the ruptures library.

    Emits :attr:`~whytrend.core.enums.AnomalyType.CHANGEPOINT` detections.
    Requires the optional extra: ``pip install 'whytrend[ruptures]'``.
    """

    def __init__(
        self,
        *,
        algorithm: AlgorithmName = "pelt",
        model: CostModel = "rbf",
        penalty: float | None = 10.0,
        n_bkps: int | None = None,
        min_points: int = 10,
        jump_window: int = 5,
        ruptures_kwargs: dict[str, Any] | None = None,
    ) -> None:
        if algorithm not in {"pelt", "binseg", "bottomup"}:
            msg = "algorithm must be one of: pelt, binseg, bottomup"
            raise ValueError(msg)
        if min_points < 3:
            msg = "min_points must be >= 3"
            raise ValueError(msg)
        if jump_window < 1:
            msg = "jump_window must be >= 1"
            raise ValueError(msg)
        if penalty is not None and penalty < 0:
            msg = "penalty must be >= 0"
            raise ValueError(msg)
        if n_bkps is not None and n_bkps < 1:
            msg = "n_bkps must be >= 1"
            raise ValueError(msg)
        if algorithm == "pelt" and penalty is None:
            msg = "pelt requires penalty"
            raise ValueError(msg)
        if algorithm != "pelt" and penalty is None and n_bkps is None:
            msg = "provide penalty and/or n_bkps"
            raise ValueError(msg)

        self._algorithm = algorithm
        self._model = model
        self._penalty = penalty
        self._n_bkps = n_bkps
        self._min_points = min_points
        self._jump_window = jump_window
        self._ruptures_kwargs = dict(ruptures_kwargs or {})

    @property
    def name(self) -> str:
        return "ruptures"

    def detect(self, series: TimeSeries) -> DetectionResult:
        values = series.data.astype(float)
        if len(values) < self._min_points:
            return self._empty_result(series, reason="insufficient_points")

        signal = values.to_numpy(dtype=float, copy=True)
        if float(np.std(signal)) == 0.0:
            return self._empty_result(series, reason="zero_variance")

        breakpoints = self._fit_detect(signal)
        change_indexes = [index for index in breakpoints if 0 < index < len(signal)]

        detections: list[Detection] = []
        series_std = float(np.std(signal)) or 1.0
        for index in change_indexes:
            timestamp = pd.Timestamp(values.index[index]).to_pydatetime()
            value = float(signal[index])
            left_mean, right_mean = self._segment_means(signal, index)
            jump = abs(right_mean - left_mean)
            score = clamp_score(jump / (series_std * 2.0))
            detections.append(
                Detection(
                    timestamp=timestamp,
                    anomaly_type=AnomalyType.CHANGEPOINT,
                    value=value,
                    score=score,
                    expected_value=left_mean,
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
                "algorithm": self._algorithm,
                "model": self._model,
                "penalty": self._penalty,
                "n_bkps": self._n_bkps,
                "break_count": len(detections),
            },
        )

    def _fit_detect(self, signal: np.ndarray) -> list[int]:
        try:
            import ruptures as rpt
        except ImportError as exc:
            msg = (
                "RupturesDetector requires ruptures; install with: pip install 'whytrend[ruptures]'"
            )
            raise ImportError(msg) from exc

        algo_cls = {
            "pelt": rpt.Pelt,
            "binseg": rpt.Binseg,
            "bottomup": rpt.BottomUp,
        }[self._algorithm]
        algorithm = algo_cls(model=self._model, **self._ruptures_kwargs).fit(signal)

        if self._algorithm == "pelt":
            penalty = self._penalty
            if penalty is None:
                msg = "pelt requires penalty"
                raise ValueError(msg)
            breakpoints = algorithm.predict(pen=penalty)
        elif self._n_bkps is not None:
            breakpoints = algorithm.predict(n_bkps=self._n_bkps)
        else:
            penalty = self._penalty
            if penalty is None:
                msg = "provide penalty and/or n_bkps"
                raise ValueError(msg)
            breakpoints = algorithm.predict(pen=penalty)

        return [int(index) for index in breakpoints]

    def _segment_means(self, signal: np.ndarray, index: int) -> tuple[float, float]:
        left_start = max(0, index - self._jump_window)
        right_end = min(len(signal), index + self._jump_window)
        left = signal[left_start:index]
        right = signal[index:right_end]
        left_mean = float(left.mean()) if len(left) else float(signal[index])
        right_mean = float(right.mean()) if len(right) else float(signal[index])
        return left_mean, right_mean

    def _empty_result(self, series: TimeSeries, *, reason: str) -> DetectionResult:
        return DetectionResult(
            detections=[],
            detector_name=self.name,
            series_name=series.name,
            metadata={
                "reason": reason,
                "algorithm": self._algorithm,
                "model": self._model,
                "penalty": self._penalty,
                "n_bkps": self._n_bkps,
                "break_count": 0,
            },
        )
