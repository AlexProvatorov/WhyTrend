from datetime import UTC, datetime

import pytest

from tests.stubs import StubCollector, StubDetector, StubExplainer, StubSource
from whytrend.core import AnomalyType, Cause, Event, Evidence, EvidenceVerdict, Explanation, Report
from whytrend.explainers import LLMExplainer
from whytrend.filters import EvidenceFilter
from whytrend.llm import MockLLMProvider
from whytrend.llm._parsing import fallback_explanation
from whytrend.pipeline import Pipeline
from whytrend.report import HTMLRenderer, JSONRenderer, MarkdownRenderer


def _event() -> Event:
    return Event(
        anomaly_type=AnomalyType.SPIKE,
        timestamp=datetime(2026, 1, 4, tzinfo=UTC),
        window_start=datetime(2026, 1, 1, tzinfo=UTC),
        window_end=datetime(2026, 1, 7, tzinfo=UTC),
        keyword="Python",
        series_name="python_interest",
        value=610.0,
        detection_score=0.95,
    )


def _evidence(title: str, score: float) -> Evidence:
    return Evidence(
        title=title,
        url=f"https://example.com/{title}",
        snippet=title,
        source_name="hacker_news",
        relevance_score=score,
    )


def test_filter_drops_items_below_keep_threshold_and_reranks() -> None:
    kept, verdict = EvidenceFilter(keep_threshold=0.2, cause_threshold=0.6).filter(
        [
            _evidence("weak", 0.19),
            _evidence("middle", 0.4),
            _evidence("strong", 0.91),
        ]
    )

    assert verdict is EvidenceVerdict.LIKELY_CAUSE
    assert [item.title for item in kept] == ["strong", "middle"]
    assert all(item.relevance_score >= 0.2 for item in kept)


def test_filter_correlation_only_when_nothing_clears_cause_threshold() -> None:
    kept, verdict = EvidenceFilter().filter([_evidence("middle", 0.45), _evidence("lower", 0.3)])

    assert verdict is EvidenceVerdict.CORRELATION_ONLY
    assert [item.title for item in kept] == ["middle", "lower"]


def test_filter_evidence_insufficient_when_nothing_is_kept() -> None:
    kept, verdict = EvidenceFilter().filter([_evidence("weak", 0.05)])

    assert kept == []
    assert verdict is EvidenceVerdict.EVIDENCE_INSUFFICIENT


def test_filter_evidence_insufficient_for_empty_input() -> None:
    assert EvidenceFilter().filter([]) == ([], EvidenceVerdict.EVIDENCE_INSUFFICIENT)


def test_filter_validation() -> None:
    with pytest.raises(ValueError, match="keep_threshold must be between 0 and 1"):
        EvidenceFilter(keep_threshold=-0.1)
    with pytest.raises(ValueError, match="cause_threshold must be >= keep_threshold"):
        EvidenceFilter(keep_threshold=0.8, cause_threshold=0.4)
    with pytest.raises(ValueError, match="correlation_only_cap must be between 0 and 1"):
        EvidenceFilter(correlation_only_cap=1.2)


@pytest.mark.asyncio
async def test_mock_verdicts_control_summary_wording() -> None:
    event = _event()
    strong = [_evidence("Python 3.13 released", 0.92)]
    medium = [_evidence("Python 3.13 released", 0.4)]

    likely = await MockLLMProvider().explain(event, strong, verdict=EvidenceVerdict.LIKELY_CAUSE)
    correlation = await MockLLMProvider().explain(
        event, medium, verdict=EvidenceVerdict.CORRELATION_ONLY
    )
    insufficient = await MockLLMProvider().explain(
        event, [], verdict=EvidenceVerdict.EVIDENCE_INSUFFICIENT
    )
    fallback = fallback_explanation(event, [], verdict=EvidenceVerdict.EVIDENCE_INSUFFICIENT)

    assert likely.summary.startswith("Likely cause")
    assert likely.verdict is EvidenceVerdict.LIKELY_CAUSE
    assert "Likely cause" not in correlation.summary
    assert correlation.summary.startswith("Correlation only")
    assert correlation.verdict is EvidenceVerdict.CORRELATION_ONLY
    assert insufficient.causes == []
    assert "Likely cause" not in insufficient.summary
    assert "insufficient" in insufficient.summary
    assert fallback.causes == []
    assert "insufficient" in fallback.summary


@pytest.mark.asyncio
async def test_pipeline_hides_weak_evidence_from_explainer(
    python_interest_series,
    spike_detection,
) -> None:
    class RecordingExplainer(StubExplainer):
        def __init__(self) -> None:
            super().__init__()
            self.seen: list[Evidence] = []

        async def explain(self, event, evidences):
            self.seen = list(evidences)
            return await super().explain(event, evidences)

    explainer = RecordingExplainer()
    pipeline = (
        Pipeline()
        .add_source(StubSource(python_interest_series))
        .add_detector(StubDetector([spike_detection]))
        .add_collector(
            StubCollector(
                [
                    _evidence("weak", 0.05),
                    _evidence("strong", 0.9),
                ]
            )
        )
        .add_filter(EvidenceFilter())
        .add_explainer(explainer)
    )

    report = await pipeline.arun()

    assert [item.title for item in explainer.seen] == ["strong"]
    assert report.explanations[0].verdict is EvidenceVerdict.LIKELY_CAUSE
    assert report.metadata["filter"] == "evidence"


@pytest.mark.asyncio
async def test_pipeline_insufficient_verdict_has_no_cause(
    python_interest_series,
    spike_detection,
) -> None:
    pipeline = (
        Pipeline()
        .add_source(StubSource(python_interest_series))
        .add_detector(StubDetector([spike_detection]))
        .add_collector(StubCollector([_evidence("weak", 0.05)]))
        .add_filter(EvidenceFilter())
        .add_explainer(LLMExplainer(MockLLMProvider()))
    )

    report = await pipeline.arun()
    explanation = report.explanations[0]

    assert explanation.verdict is EvidenceVerdict.EVIDENCE_INSUFFICIENT
    assert explanation.causes == []
    assert "Likely cause" not in explanation.summary
    assert "insufficient" in explanation.summary
    assert "insufficient" in report.executive_summary


@pytest.mark.asyncio
async def test_pipeline_without_filter_leaves_verdict_unset(
    python_interest_series,
    spike_detection,
    sample_evidence,
) -> None:
    pipeline = (
        Pipeline()
        .add_source(StubSource(python_interest_series))
        .add_detector(StubDetector([spike_detection]))
        .add_collector(StubCollector([sample_evidence]))
        .add_explainer(StubExplainer())
    )

    report = await pipeline.arun()

    assert report.explanations[0].verdict is None
    assert report.explanations[0].summary == "Explained Python spike."
    assert report.explanations[0].confidence == 0.95
    assert "confidence_capped" not in report.explanations[0].metadata
    assert report.metadata["filter"] is None


def test_renderers_print_verdict() -> None:
    explanation = Explanation(
        event_id="evt-1",
        anomaly_type=AnomalyType.SPIKE,
        confidence=0.4,
        verdict=EvidenceVerdict.CORRELATION_ONLY,
        summary="Correlation only: interest in 'Python' spiked due to a headline.",
    )
    report = Report(
        executive_summary=explanation.summary,
        explanations=[explanation],
        series_name="python_interest",
        keyword="Python",
    )

    assert "- **Verdict:** correlation_only" in MarkdownRenderer().render(report)
    assert "<strong>Verdict:</strong> correlation_only" in HTMLRenderer().render(report)
    restored = Report.model_validate_json(JSONRenderer().render(report))
    assert restored.explanations[0].verdict is EvidenceVerdict.CORRELATION_ONLY


@pytest.mark.asyncio
async def test_pipeline_correlation_summary_does_not_say_likely_cause(
    python_interest_series,
    spike_detection,
) -> None:
    pipeline = (
        Pipeline()
        .add_source(StubSource(python_interest_series))
        .add_detector(StubDetector([spike_detection]))
        .add_collector(StubCollector([_evidence("Python weekly", 0.4)]))
        .add_filter(EvidenceFilter())
        .add_explainer(LLMExplainer(MockLLMProvider()))
    )

    explanation = (await pipeline.arun()).explanations[0]

    assert explanation.verdict is EvidenceVerdict.CORRELATION_ONLY
    assert "Likely cause" not in explanation.summary
    assert explanation.summary.startswith("Correlation only")


def test_duplicate_filter_raises(pipeline_components) -> None:
    source, detector, _, explainer = pipeline_components
    pipeline = Pipeline().add_source(source).add_detector(detector).add_filter(EvidenceFilter())

    with pytest.raises(ValueError, match="filter is already configured"):
        pipeline.add_filter(EvidenceFilter())

    pipeline.add_explainer(explainer)


def _overconfident(verdict: EvidenceVerdict | None) -> Explanation:
    return Explanation(
        event_id="evt-1",
        anomaly_type=AnomalyType.SPIKE,
        confidence=0.95,
        verdict=verdict,
        summary="Likely cause: interest in 'Python' spiked due to a headline.",
        causes=[
            Cause(
                source="hacker_news",
                score=0.9,
                url="https://example.com/story",
                title="A headline",
                summary="details",
            )
        ],
    )


def test_cap_uses_the_verdict_not_the_summary() -> None:
    evidence_filter = EvidenceFilter()
    untouched = _overconfident(None)

    assert evidence_filter.cap_confidence(untouched) is untouched

    likely = evidence_filter.cap_confidence(_overconfident(EvidenceVerdict.LIKELY_CAUSE))
    assert likely.confidence == 0.95
    assert likely.causes[0].score == 0.9
    assert "confidence_capped" not in likely.metadata

    correlation = evidence_filter.cap_confidence(_overconfident(EvidenceVerdict.CORRELATION_ONLY))
    assert correlation.confidence == 0.6
    assert correlation.causes[0].score == 0.6
    assert correlation.metadata["confidence_capped"] is True
    assert correlation.summary.startswith("Likely cause")

    custom = EvidenceFilter(correlation_only_cap=0.5).cap_confidence(
        _overconfident(EvidenceVerdict.CORRELATION_ONLY)
    )
    assert custom.confidence == 0.5
    assert custom.causes[0].score == 0.5

    insufficient = evidence_filter.cap_confidence(
        _overconfident(EvidenceVerdict.EVIDENCE_INSUFFICIENT)
    )
    assert insufficient.confidence == 0.3
    assert insufficient.causes[0].score == 0.3
    assert insufficient.metadata["confidence_capped"] is True


def test_cap_leaves_values_that_are_already_under_the_ceiling() -> None:
    explanation = Explanation(
        event_id="evt-1",
        anomaly_type=AnomalyType.SPIKE,
        confidence=0.4,
        verdict=EvidenceVerdict.CORRELATION_ONLY,
        summary="Correlation only.",
        causes=[
            Cause(
                source="hacker_news",
                score=0.2,
                url="https://example.com/story",
                title="A headline",
            )
        ],
    )

    assert EvidenceFilter().cap_confidence(explanation) is explanation


def test_capped_confidence_is_what_the_report_renders() -> None:
    capped = EvidenceFilter().cap_confidence(_overconfident(EvidenceVerdict.CORRELATION_ONLY))
    report = Report(
        executive_summary=capped.summary,
        explanations=[capped],
        series_name="python_interest",
        keyword="Python",
    )

    markdown = MarkdownRenderer().render(report)
    html = HTMLRenderer().render(report)
    assert "- **Confidence:** 0.60" in markdown
    assert "(0.60)" in markdown
    assert "0.95" not in markdown
    assert "<strong>Confidence:</strong> 0.60" in html
    restored = Report.model_validate_json(JSONRenderer().render(report))
    assert restored.explanations[0].confidence == 0.6
    assert restored.explanations[0].causes[0].score == 0.6


class _OverconfidentExplainer(StubExplainer):
    async def explain(self, event, evidences):
        explanation = await super().explain(event, evidences)
        source = evidences[0] if evidences else None
        cause = Cause(
            source=source.source_name if source is not None else "hacker_news",
            score=0.9,
            url=source.url if source is not None else "https://example.com/story",
            title=source.title if source is not None else "A headline",
            summary="details",
        )
        return explanation.model_copy(update={"confidence": 0.95, "causes": [cause]})


@pytest.mark.asyncio
async def test_pipeline_caps_correlation_and_insufficient_confidence(
    python_interest_series,
    spike_detection,
) -> None:
    async def run(score: float) -> Explanation:
        pipeline = (
            Pipeline()
            .add_source(StubSource(python_interest_series))
            .add_detector(StubDetector([spike_detection]))
            .add_collector(StubCollector([_evidence("Python weekly", score)]))
            .add_filter(EvidenceFilter())
            .add_explainer(_OverconfidentExplainer())
        )
        return (await pipeline.arun()).explanations[0]

    correlation = await run(0.4)
    assert correlation.verdict is EvidenceVerdict.CORRELATION_ONLY
    assert correlation.confidence == 0.6
    assert correlation.causes[0].score == 0.6
    assert correlation.metadata["confidence_capped"] is True
    rendered = MarkdownRenderer().render(
        Report(
            executive_summary=correlation.summary,
            explanations=[correlation],
            keyword="Python",
        )
    )
    assert "- **Confidence:** 0.60" in rendered

    insufficient = await run(0.05)
    assert insufficient.verdict is EvidenceVerdict.EVIDENCE_INSUFFICIENT
    assert insufficient.confidence == 0.3
    assert insufficient.causes == []
    assert insufficient.metadata["confidence_capped"] is True

    likely = await run(0.9)
    assert likely.verdict is EvidenceVerdict.LIKELY_CAUSE
    assert likely.confidence == 0.95
    assert likely.causes[0].score == 0.9
    assert "confidence_capped" not in likely.metadata
