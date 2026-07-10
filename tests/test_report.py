from datetime import datetime, timezone

import pytest

from whytrend.core import AnomalyType, Cause, Explanation, Report
from whytrend.report import JSONRenderer, MarkdownRenderer


def _sample_report() -> Report:
    explanation = Explanation(
        event_id="evt-1",
        anomaly_type=AnomalyType.SPIKE,
        confidence=0.94,
        summary="Interest in 'Python' spiked due to the release of Python 3.13.",
        causes=[
            Cause(
                source="hacker_news",
                score=0.92,
                url="https://example.com/python-3-13",
                title="Python 3.13 released",
                summary="Release announcement.",
            )
        ],
    )
    return Report(
        executive_summary=explanation.summary,
        explanations=[explanation],
        series_name="python_interest",
        keyword="Python",
        generated_at=datetime(2026, 1, 4, 12, 0, tzinfo=timezone.utc),
        metadata={"detector": "zscore", "source": "csv"},
    )


def test_json_renderer_returns_valid_json_round_trip() -> None:
    report = _sample_report()
    rendered = JSONRenderer().render(report)

    restored = Report.model_validate_json(rendered)

    assert restored.executive_summary == report.executive_summary
    assert restored.explanations[0].causes[0].url == report.explanations[0].causes[0].url


def test_markdown_renderer_includes_summary_and_causes() -> None:
    report = _sample_report()
    rendered = MarkdownRenderer().render(report)

    assert "# WhyTrend Report" in rendered
    assert "## Executive Summary" in rendered
    assert "Python 3.13 released" in rendered
    assert "https://example.com/python-3-13" in rendered
    assert "**Keyword:** Python" in rendered


def test_report_convenience_methods_match_renderers() -> None:
    report = _sample_report()

    assert report.to_json() == JSONRenderer().render(report)
    assert report.to_markdown() == MarkdownRenderer().render(report)


def test_markdown_renderer_handles_empty_explanations() -> None:
    report = Report(
        executive_summary="No anomalies detected.",
        series_name="python_interest",
        keyword="Python",
    )

    rendered = MarkdownRenderer().render(report)

    assert "_No explanations were generated._" in rendered


def test_json_renderer_validation() -> None:
    with pytest.raises(ValueError, match="indent must be >= 0"):
        JSONRenderer(indent=-1)
