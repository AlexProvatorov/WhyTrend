import pytest

from whytrend.core import AnomalyType
from whytrend.pipeline import Pipeline
from tests.stubs import FailingCollector, StubCollector, StubDetector, StubExplainer, StubSource


@pytest.mark.asyncio
async def test_arun_returns_report_with_explanation(configured_pipeline) -> None:
    report = await configured_pipeline.arun()

    assert report.keyword == "Python"
    assert report.series_name == "python_interest"
    assert report.executive_summary == "Explained Python spike."
    assert len(report.explanations) == 1
    assert report.explanations[0].anomaly_type is AnomalyType.SPIKE
    assert report.explanations[0].causes[0].source == "Google News"
    assert report.metadata["event_count"] == 1


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


def test_run_requires_source() -> None:
    with pytest.raises(ValueError, match="source is required"):
        Pipeline().run()
