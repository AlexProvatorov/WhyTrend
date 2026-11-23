from datetime import UTC, datetime

import pytest

from tests.stubs import (
    FailingCollector,
    FailingExplainer,
    StubCollector,
    StubDetector,
    StubExplainer,
    StubSource,
)
from whytrend.core import AnomalyType
from whytrend.pipeline import Pipeline
from whytrend.rankers import BM25Ranker


@pytest.mark.asyncio
async def test_arun_returns_report_with_explanation(configured_pipeline) -> None:
    report = await configured_pipeline.arun()

    assert report.keyword == "Python"
    assert report.series_name == "python_interest"
    assert report.executive_summary == "Explained Python spike."
    assert len(report.explanations) == 1
    explanation = report.explanations[0]
    assert explanation.anomaly_type is AnomalyType.SPIKE
    assert explanation.causes[0].source == "Google News"
    assert explanation.timestamp is not None
    assert explanation.value is not None
    assert explanation.window_start is not None
    assert explanation.window_end is not None
    assert report.metadata["event_count"] == 1


@pytest.mark.asyncio
async def test_executive_summary_separates_events(
    python_interest_series,
    spike_detection,
    sample_evidence,
) -> None:
    from whytrend.core.models import Detection
    from whytrend.report import HTMLRenderer, MarkdownRenderer

    drop = Detection(
        timestamp=datetime(2026, 1, 2, tzinfo=UTC),
        anomaly_type=AnomalyType.DROP,
        value=40.0,
        score=0.8,
        expected_value=120.0,
    )
    pipeline = (
        Pipeline()
        .add_source(StubSource(python_interest_series))
        .add_detector(StubDetector([spike_detection, drop]))
        .add_collector(StubCollector([sample_evidence]))
        .add_explainer(StubExplainer())
    )

    report = await pipeline.arun()

    assert report.executive_summary == ("Explained Python drop.\n\nExplained Python spike.")
    markdown = MarkdownRenderer().render(report)
    assert "Explained Python drop.\n\nExplained Python spike." in markdown
    executive, _, _rest = HTMLRenderer().render(report).partition("<h2>Explanations</h2>")
    assert executive.count("<p>Explained Python") == 2


def test_merged_detections_reach_the_report(python_interest_series) -> None:
    from whytrend.detectors import ZScoreDetector
    from whytrend.report import HTMLRenderer, MarkdownRenderer
    from whytrend.sources import PandasSource

    report = (
        Pipeline()
        .add_source(
            PandasSource(
                python_interest_series.to_dataframe(),
                name="python_interest",
                keyword="Python",
            )
        )
        .add_detector(ZScoreDetector())
        .add_explainer(StubExplainer())
        .run()
    )

    assert report.metadata["event_count"] == 1
    assert report.metadata["merged_detections"] == [2]
    assert report.explanations[0].metadata["merged_detections"] == 2
    assert "Merged detections:** 2" in MarkdownRenderer().render(report)
    assert "Merged detections:</strong> 2" in HTMLRenderer().render(report)


def test_run_executes_pipeline_synchronously(configured_pipeline) -> None:
    report = configured_pipeline.run()

    assert report.executive_summary.startswith("Explained Python")


def test_fluent_api_returns_pipeline_instance(pipeline_components) -> None:
    source, detector, collector, explainer = pipeline_components
    pipeline = Pipeline()

    assert pipeline.add_source(source) is pipeline
    assert pipeline.add_detector(detector) is pipeline
    assert pipeline.add_collector(collector) is pipeline
    assert pipeline.add_explainer(explainer) is pipeline


def test_duplicate_source_raises(pipeline_components) -> None:
    source, detector, _, explainer = pipeline_components
    pipeline = Pipeline().add_source(source)

    with pytest.raises(ValueError, match="source is already configured"):
        pipeline.add_source(source)

    pipeline.add_detector(detector).add_explainer(explainer)


@pytest.mark.asyncio
async def test_arun_without_detections_returns_empty_report(python_interest_series) -> None:
    pipeline = (
        Pipeline()
        .add_source(StubSource(python_interest_series))
        .add_detector(StubDetector([]))
        .add_explainer(StubExplainer())
    )

    report = await pipeline.arun()

    assert report.executive_summary == "No anomalies detected."
    assert report.explanations == []
    assert report.metadata["event_count"] == 0


@pytest.mark.asyncio
async def test_failing_collector_does_not_break_pipeline(
    python_interest_series,
    spike_detection,
    sample_evidence,
) -> None:
    pipeline = (
        Pipeline()
        .add_source(StubSource(python_interest_series))
        .add_detector(StubDetector([spike_detection]))
        .add_collector(FailingCollector())
        .add_collector(StubCollector([sample_evidence], name="backup-collector"))
        .add_explainer(StubExplainer())
    )

    report = await pipeline.arun()

    assert len(report.explanations) == 1
    assert report.explanations[0].metadata["evidence_count"] == 1
    errors = report.metadata["collector_errors"]
    assert errors[0]["collector"] == "failing-collector"
    assert errors[0]["error"] == "RuntimeError"
    assert errors[0]["message"] == "collector unavailable"
    assert report.explanations[0].metadata["collector_errors"] == errors


@pytest.mark.asyncio
async def test_failing_explainer_falls_back_without_breaking_pipeline(
    python_interest_series,
    spike_detection,
    sample_evidence,
) -> None:
    pipeline = (
        Pipeline()
        .add_source(StubSource(python_interest_series))
        .add_detector(StubDetector([spike_detection]))
        .add_collector(StubCollector([sample_evidence]))
        .add_explainer(FailingExplainer())
    )

    report = await pipeline.arun()

    assert len(report.explanations) == 1
    assert report.explanations[0].metadata.get("fallback") is True
    assert report.explanations[0].metadata.get("explainer_error") == "RuntimeError"
    assert report.explanations[0].causes[0].url == sample_evidence.url


def test_duplicate_ranker_raises(pipeline_components) -> None:
    source, detector, collector, explainer = pipeline_components
    pipeline = Pipeline().add_source(source).add_detector(detector).add_ranker(BM25Ranker())

    with pytest.raises(ValueError, match="ranker is already configured"):
        pipeline.add_ranker(BM25Ranker())

    pipeline.add_collector(collector).add_explainer(explainer)


@pytest.mark.asyncio
async def test_run_works_inside_a_running_event_loop(configured_pipeline) -> None:
    report = configured_pipeline.run()

    assert report.keyword == "Python"


@pytest.mark.asyncio
async def test_empty_report_names_the_detector_reason(python_interest_series) -> None:
    from datetime import UTC, datetime

    import pandas as pd

    from whytrend.core.series import TimeSeries
    from whytrend.detectors import ZScoreDetector

    short = python_interest_series.window(
        datetime(2026, 1, 1, tzinfo=UTC),
        datetime(2026, 1, 3, tzinfo=UTC),
    )
    short_report = await (
        Pipeline()
        .add_source(StubSource(short))
        .add_detector(ZScoreDetector(min_points=5))
        .add_explainer(StubExplainer())
    ).arun()

    assert short_report.metadata["reason"] == "insufficient_points"
    assert "insufficient points" in short_report.executive_summary

    frame = pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-01-01", periods=6, freq="D", tz="UTC"),
            "value": [10.0] * 6,
        }
    )
    flat = TimeSeries.from_dataframe(frame, time_col="timestamp", value_col="value", name="flat")
    flat_report = await (
        Pipeline()
        .add_source(StubSource(flat, name="flat"))
        .add_detector(ZScoreDetector())
        .add_explainer(StubExplainer())
    ).arun()

    assert flat_report.metadata["reason"] == "zero_variance"
    assert "zero" in flat_report.executive_summary


@pytest.mark.asyncio
async def test_invented_cause_url_is_dropped(
    python_interest_series,
    spike_detection,
    sample_evidence,
) -> None:
    from whytrend.core import Cause

    class _InventingExplainer(StubExplainer):
        async def explain(self, event, evidences):
            explanation = await super().explain(event, evidences)
            invented = Cause(
                source="blog",
                score=0.99,
                url="https://example.com/not-collected",
                title="Invented",
            )
            return explanation.model_copy(update={"causes": [*explanation.causes, invented]})

    pipeline = (
        Pipeline()
        .add_source(StubSource(python_interest_series))
        .add_detector(StubDetector([spike_detection]))
        .add_collector(StubCollector([sample_evidence]))
        .add_explainer(_InventingExplainer())
    )

    explanation = (await pipeline.arun()).explanations[0]

    assert [cause.url for cause in explanation.causes] == [sample_evidence.url]


@pytest.mark.asyncio
async def test_explainer_failure_keeps_the_message_and_raw_response(
    python_interest_series,
    spike_detection,
    sample_evidence,
) -> None:
    from whytrend.llm._parsing import LLMResponseError

    class _BadModel:
        @property
        def name(self) -> str:
            return "bad-model"

        async def explain(self, event, evidences):
            _ = event, evidences
            msg = "LLM response was not valid JSON"
            raise LLMResponseError(msg, raw_response='{"response_schema": {"summary": 1}}')

    pipeline = (
        Pipeline()
        .add_source(StubSource(python_interest_series))
        .add_detector(StubDetector([spike_detection]))
        .add_collector(StubCollector([sample_evidence]))
        .add_explainer(_BadModel())
    )

    metadata = (await pipeline.arun()).explanations[0].metadata

    assert metadata["explainer_error"] == "LLMResponseError"
    assert "not valid JSON" in metadata["explainer_error_message"]
    assert "response_schema" in metadata["explainer_raw_response"]


def test_run_requires_source() -> None:
    with pytest.raises(ValueError, match="source is required"):
        Pipeline().run()
