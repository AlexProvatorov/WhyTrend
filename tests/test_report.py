from datetime import UTC, datetime
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from whytrend.core import AnomalyType, Cause, Explanation, Report
from whytrend.report import HTMLRenderer, JSONRenderer, MarkdownRenderer, PDFRenderer


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
        generated_at=datetime(2026, 1, 4, 12, 0, tzinfo=UTC),
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
    assert report.to_html() == HTMLRenderer().render(report)


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


def test_html_renderer_includes_summary_causes_and_urls() -> None:
    report = _sample_report()
    rendered = HTMLRenderer().render(report)

    assert HTMLRenderer().name == "html"
    assert "<!DOCTYPE html>" in rendered
    assert "Executive Summary" in rendered
    assert "Python 3.13 released" in rendered
    assert "https://example.com/python-3-13" in rendered
    assert 'href="https://example.com/python-3-13"' in rendered
    assert "Python" in rendered
    assert "zscore" in rendered


def test_html_renderer_handles_empty_explanations() -> None:
    report = Report(
        executive_summary="No anomalies detected.",
        series_name="python_interest",
        keyword="Python",
    )

    rendered = HTMLRenderer().render(report)

    assert "No explanations were generated." in rendered


def test_html_renderer_escapes_untrusted_text() -> None:
    report = Report(
        executive_summary="<script>alert(1)</script>",
        series_name="<b>series</b>",
        keyword="Python",
    )

    rendered = HTMLRenderer().render(report)

    assert "<script>alert(1)</script>" not in rendered
    assert "&lt;script&gt;alert(1)&lt;/script&gt;" in rendered
    assert "<b>series</b>" not in rendered


def test_pdf_renderer_name_and_import_without_extra() -> None:
    assert PDFRenderer().name == "pdf"


def test_pdf_renderer_returns_bytes_when_converter_is_mocked(tmp_path) -> None:
    report = _sample_report()
    payload = b"%PDF-1.4 fake"

    class FakePisa:
        @staticmethod
        def CreatePDF(src, dest, encoding="utf-8"):
            _ = src, encoding
            dest.write(payload)
            return SimpleNamespace(err=0)

    with patch.object(PDFRenderer, "_load_pisa", return_value=FakePisa):
        rendered = PDFRenderer().render(report)
        via_helper = report.to_pdf()

    assert rendered == payload
    assert via_helper == payload
    out = tmp_path / "report.pdf"
    out.write_bytes(rendered)
    assert out.read_bytes() == payload


def test_pdf_renderer_requires_optional_extra() -> None:
    with (
        patch.object(PDFRenderer, "_load_pisa", side_effect=ImportError("whytrend[pdf]")),
        pytest.raises(ImportError, match=r"whytrend\[pdf\]"),
    ):
        PDFRenderer().render(_sample_report())


def test_pdf_renderer_real_conversion_if_extra_installed() -> None:
    pytest.importorskip("xhtml2pdf")
    rendered = PDFRenderer().render(_sample_report())
    assert rendered.startswith(b"%PDF")
    assert len(rendered) > 0
