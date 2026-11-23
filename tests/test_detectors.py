import logging
import sys
import types
from datetime import UTC, datetime, timedelta
from unittest.mock import patch

import pandas as pd
import pytest

from tests.stubs import StubCollector, StubExplainer
from whytrend.core import AnomalyType
from whytrend.core.series import TimeSeries
from whytrend.detectors import ProphetDetector, RiverDetector, RupturesDetector, ZScoreDetector
from whytrend.events import EventBuilder
from whytrend.pipeline import Pipeline
from whytrend.sources import PandasSource


def test_zscore_detector_finds_spike(python_interest_series) -> None:
    detector = ZScoreDetector(threshold=1.0)
    result = detector.detect(python_interest_series)

    assert result.detector_name == "zscore"
    assert result.series_name == "python_interest"
    assert result.primary is not None
    assert result.primary.anomaly_type is AnomalyType.SPIKE
    assert result.primary.value == 640.0
    assert result.primary.expected_value == pytest.approx(123.0)
    assert result.primary.score > 0.0


def test_zscore_default_threshold_sees_the_fixture_spike(python_interest_series) -> None:
    result = ZScoreDetector().detect(python_interest_series)

    assert result.metadata.get("reason") is None
    assert {item.value for item in result.detections} == {610.0, 640.0}
    events = EventBuilder().build(result, python_interest_series)
    assert len(events) == 1
    assert events[0].value == 640.0
    assert events[0].metadata["merged_detections"] == 2


def test_zscore_detector_returns_empty_for_short_series(python_interest_series) -> None:
    short_series = python_interest_series.window(
        datetime(2026, 1, 1, tzinfo=UTC),
        datetime(2026, 1, 3, tzinfo=UTC),
    )

    result = ZScoreDetector(min_points=5).detect(short_series)

    assert result.detections == []
    assert result.metadata["reason"] == "insufficient_points"


def test_zscore_detector_rejects_invalid_threshold() -> None:
    with pytest.raises(ValueError, match="threshold must be > 0"):
        ZScoreDetector(threshold=0)


def test_prophet_detector_finds_points_outside_forecast_band(python_interest_series) -> None:
    frame = python_interest_series.to_dataframe().rename(columns={"timestamp": "ds", "value": "y"})
    forecast = pd.DataFrame(
        {
            "ds": frame["ds"],
            "yhat": [122.0] * len(frame),
            "yhat_lower": [100.0] * len(frame),
            "yhat_upper": [140.0] * len(frame),
        }
    )

    detector = ProphetDetector(min_points=5)
    with patch.object(detector, "_fit_and_predict", return_value=forecast):
        result = detector.detect(python_interest_series)

    assert result.detector_name == "prophet"
    assert result.primary is not None
    assert result.primary.value in {610.0, 640.0}
    assert result.primary.anomaly_type is AnomalyType.SPIKE


def test_prophet_fits_naive_utc_and_hides_missing_plotly(
    python_interest_series,
    caplog: pytest.LogCaptureFixture,
) -> None:
    captured: dict[str, object] = {}

    class _Prophet:
        def __init__(self, **kwargs: object) -> None:
            captured["kwargs"] = kwargs

        def fit(self, frame: pd.DataFrame) -> None:
            captured["fit_tz"] = frame["ds"].dt.tz

        def predict(self, frame: pd.DataFrame) -> pd.DataFrame:
            captured["predict_tz"] = frame["ds"].dt.tz
            count = len(frame)
            return pd.DataFrame(
                {
                    "ds": frame["ds"],
                    "yhat": [100.0] * count,
                    "yhat_lower": [90.0] * count,
                    "yhat_upper": [110.0] * count,
                }
            )

    class _ProphetModule(types.ModuleType):
        @property
        def Prophet(self) -> type[_Prophet]:
            logging.getLogger("prophet.plot").error(
                "Importing plotly failed. Interactive plots will not work."
            )
            return _Prophet

    with caplog.at_level(logging.ERROR):
        with patch.dict(sys.modules, {"prophet": _ProphetModule("prophet")}):
            result = ProphetDetector(min_points=5).detect(python_interest_series)
        logging.getLogger("prophet").error("Stan failed to fit")

    assert captured["fit_tz"] is None
    assert captured["predict_tz"] is None
    assert result.detections
    assert all(item.timestamp.utcoffset() == timedelta(0) for item in result.detections)
    assert "Importing plotly failed" not in caplog.text
    assert "Stan failed to fit" in caplog.text


def test_ruptures_detector_emits_changepoints(python_interest_series) -> None:
    detector = RupturesDetector(min_points=5, penalty=1.0)
    # ruptures returns segment ends; last index is series length and is ignored.
    with patch.object(detector, "_fit_detect", return_value=[3, len(python_interest_series)]):
        result = detector.detect(python_interest_series)

    assert detector.name == "ruptures"
    assert result.detector_name == "ruptures"
    assert len(result.detections) == 1
    assert result.primary is not None
    assert result.primary.anomaly_type is AnomalyType.CHANGEPOINT
    assert result.primary.index == 3
    assert result.metadata["algorithm"] == "pelt"
    assert result.metadata["break_count"] == 1


def test_ruptures_detector_returns_empty_for_short_series(python_interest_series) -> None:
    short_series = python_interest_series.window(
        datetime(2026, 1, 1, tzinfo=UTC),
        datetime(2026, 1, 3, tzinfo=UTC),
    )

    result = RupturesDetector(min_points=5).detect(short_series)

    assert result.detections == []
    assert result.metadata["reason"] == "insufficient_points"
    assert result.metadata["break_count"] == 0


def test_ruptures_detector_returns_empty_for_zero_variance() -> None:
    frame = pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-01-01", periods=12, freq="D", tz="UTC"),
            "value": [10.0] * 12,
        }
    )
    series = TimeSeries.from_dataframe(
        frame,
        time_col="timestamp",
        value_col="value",
        name="flat",
    )
    result = RupturesDetector(min_points=5).detect(series)

    assert result.detections == []
    assert result.metadata["reason"] == "zero_variance"


def test_ruptures_detector_validation() -> None:
    with pytest.raises(ValueError, match="algorithm must be one of"):
        RupturesDetector(algorithm="unknown")  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="pelt requires penalty"):
        RupturesDetector(algorithm="pelt", penalty=None, n_bkps=2)

    with pytest.raises(ValueError, match="provide penalty and/or n_bkps"):
        RupturesDetector(algorithm="binseg", penalty=None, n_bkps=None)


def test_river_detector_finds_injected_anomaly(python_interest_series) -> None:
    class _AlwaysOne:
        def score_one(self, *args: object, **kwargs: object) -> float:
            _ = args, kwargs
            return 1.0

        def learn_one(self, *args: object, **kwargs: object) -> None:
            _ = args, kwargs

    class _Anomaly:
        def GaussianScorer(self, **kwargs: object) -> _AlwaysOne:
            _ = kwargs
            return _AlwaysOne()

    with patch("whytrend.detectors.river_detector._load_river_anomaly", return_value=_Anomaly()):
        result = RiverDetector(min_points=2, threshold=0.95).detect(python_interest_series)

    assert result.detector_name == "river"
    assert result.primary is not None
    assert {item.value for item in result.detections} == {610.0, 640.0}
    assert result.primary.anomaly_type is AnomalyType.SPIKE
    assert result.metadata["model"] == "gaussian"


def test_river_detector_returns_empty_for_short_series(python_interest_series) -> None:
    short_series = python_interest_series.window(
        datetime(2026, 1, 1, tzinfo=UTC),
        datetime(2026, 1, 3, tzinfo=UTC),
    )

    result = RiverDetector(min_points=5).detect(short_series)

    assert result.detections == []
    assert result.metadata["reason"] == "insufficient_points"


def test_river_detector_update_respects_warmup_and_threshold() -> None:
    detector = RiverDetector(min_points=2, threshold=0.99)

    class _AlwaysOne:
        def score_one(self, *args: object, **kwargs: object) -> float:
            _ = args, kwargs
            return 1.0

        def learn_one(self, *args: object, **kwargs: object) -> None:
            _ = args, kwargs

    with patch.object(detector, "_create_model", return_value=_AlwaysOne()):
        assert detector.update(10.0) is None
        assert detector.update(11.0) is None
        assert detector.update(12.0) is None
        detection = detector.update(100.0)

    assert detection is not None
    assert detection.value == 100.0
    assert detection.score == 1.0


def test_river_gaussian_ignores_ordinary_noise() -> None:
    class _AlwaysOne:
        def score_one(self, *args: object, **kwargs: object) -> float:
            _ = args, kwargs
            return 1.0

        def learn_one(self, *args: object, **kwargs: object) -> None:
            _ = args, kwargs

    class _Anomaly:
        def GaussianScorer(self, **kwargs: object) -> _AlwaysOne:
            _ = kwargs
            return _AlwaysOne()

    values = [9.6 if index % 2 == 0 else 10.4 for index in range(40)]
    values.extend([9.07, 80.0])
    frame = pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-01-01", periods=len(values), freq="D", tz="UTC"),
            "value": values,
        }
    )
    series = TimeSeries.from_dataframe(
        frame,
        time_col="timestamp",
        value_col="value",
        name="noisy",
    )

    with patch("whytrend.detectors.river_detector._load_river_anomaly", return_value=_Anomaly()):
        result = RiverDetector(min_points=10, threshold=0.95).detect(series)

    assert [item.value for item in result.detections] == [80.0]


def test_river_gaussian_plateau_does_not_outrank_the_spike() -> None:
    class _AlwaysOne:
        def score_one(self, *args: object, **kwargs: object) -> float:
            _ = args, kwargs
            return 1.0

        def learn_one(self, *args: object, **kwargs: object) -> None:
            _ = args, kwargs

    class _Anomaly:
        def GaussianScorer(self, **kwargs: object) -> _AlwaysOne:
            _ = kwargs
            return _AlwaysOne()

    frame = pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-01-01", periods=41, freq="D", tz="UTC"),
            "value": [10.0] * 40 + [500.0],
        }
    )
    series = TimeSeries.from_dataframe(
        frame,
        time_col="timestamp",
        value_col="value",
        name="plateau",
    )

    with patch("whytrend.detectors.river_detector._load_river_anomaly", return_value=_Anomaly()):
        result = RiverDetector(min_points=10, threshold=0.95).detect(series)

    assert [item.value for item in result.detections] == [500.0]
    assert result.detections[0].score == 1.0


def test_river_constant_series_is_zero_variance() -> None:
    frame = pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-01-01", periods=30, freq="D", tz="UTC"),
            "value": [10.0] * 30,
        }
    )
    series = TimeSeries.from_dataframe(frame, time_col="timestamp", value_col="value", name="flat")

    result = RiverDetector(min_points=5).detect(series)

    assert result.detections == []
    assert result.metadata["reason"] == "zero_variance"


def test_river_passes_grace_period_and_series_limits() -> None:
    class _Silent:
        def score_one(self, *args, **kwargs) -> float:
            _ = args, kwargs
            return 0.0

        def learn_one(self, *args, **kwargs) -> None:
            _ = args, kwargs

    class _Anomaly:
        def __init__(self) -> None:
            self.gaussian: dict = {}
            self.trees: dict = {}

        def GaussianScorer(self, **kwargs) -> _Silent:
            self.gaussian = kwargs
            return _Silent()

        def HalfSpaceTrees(self, **kwargs) -> _Silent:
            self.trees = kwargs
            return _Silent()

    anomaly = _Anomaly()
    frame = pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-01-01", periods=12, freq="D", tz="UTC"),
            "value": [
                120.0,
                121.0,
                119.0,
                122.0,
                118.0,
                123.0,
                117.0,
                124.0,
                116.0,
                125.0,
                130.0,
                900.0,
            ],
        }
    )
    series = TimeSeries.from_dataframe(frame, time_col="timestamp", value_col="value", name="wide")

    with patch("whytrend.detectors.river_detector._load_river_anomaly", return_value=anomaly):
        RiverDetector(min_points=4, threshold=0.99).detect(series)
        RiverDetector(model="half_space_trees", min_points=4, threshold=0.99).detect(series)

    assert anomaly.gaussian["grace_period"] == 4
    assert anomaly.trees["limits"] == {"value": (116.0, 900.0)}


def test_river_detector_validation() -> None:
    with pytest.raises(ValueError, match="model must be one of"):
        RiverDetector(model="unknown")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="threshold must be in"):
        RiverDetector(threshold=0)
    with pytest.raises(ValueError, match="min_points must be >= 1"):
        RiverDetector(min_points=0)


def test_event_builder_uses_detector_output(python_interest_series) -> None:
    result = ZScoreDetector(threshold=1.0).detect(python_interest_series)
    events = EventBuilder(window_days=2).build(result, python_interest_series)

    assert len(events) >= 1
    assert events[0].keyword == "Python"
    assert events[0].detection_score == result.primary.score


@pytest.mark.asyncio
async def test_pipeline_with_zscore_detector_and_pandas_source(
    spike_dataframe,
    sample_evidence,
) -> None:
    pipeline = (
        Pipeline(window_days=2)
        .add_source(
            PandasSource(
                spike_dataframe,
                name="python_interest",
                keyword="Python",
            )
        )
        .add_detector(ZScoreDetector(threshold=1.0))
        .add_collector(StubCollector([sample_evidence]))
        .add_explainer(StubExplainer())
    )

    report = await pipeline.arun()

    assert report.keyword == "Python"
    assert report.executive_summary.startswith("Explained Python")
    assert len(report.explanations) >= 1
    assert report.metadata["detector"] == "zscore"
    assert report.metadata["source"] == "pandas"
