"""HTML report renderer."""

from __future__ import annotations

import json
from html import escape

from whytrend.core.models import Cause, Explanation, Report
from whytrend.core.protocols import BaseReportRenderer

_PAGE_CSS = """
body { font-family: Georgia, serif; max-width: 48rem; margin: 2rem auto; padding: 0 1rem;
       color: #1a1a1a; line-height: 1.5; }
h1 { font-size: 1.6rem; margin-bottom: 0.25rem; }
h2 { font-size: 1.2rem; margin-top: 1.75rem; border-bottom: 1px solid #ddd;
     padding-bottom: 0.25rem; }
h3 { font-size: 1.05rem; margin-top: 1.25rem; }
.meta { color: #555; margin: 0; padding-left: 1.25rem; }
.empty { font-style: italic; color: #666; }
.cause { margin-bottom: 0.85rem; }
.cause-title { font-weight: bold; }
a { color: #0b57d0; }
pre { background: #f5f5f5; padding: 0.75rem; overflow-x: auto; font-size: 0.85rem; }
""".strip()


class HTMLRenderer(BaseReportRenderer):
    """Render a :class:`Report` as a self-contained HTML page."""

    @property
    def name(self) -> str:
        return "html"

    def render(self, report: Report) -> str:
        parts = [
            "<!DOCTYPE html>",
            '<html lang="en">',
            "<head>",
            '<meta charset="utf-8">',
            "<title>WhyTrend Report</title>",
            f"<style>{_PAGE_CSS}</style>",
            "</head>",
            "<body>",
            "<h1>WhyTrend Report</h1>",
            '<ul class="meta">',
            f"<li><strong>Series:</strong> {escape(report.series_name or 'n/a')}</li>",
            f"<li><strong>Keyword:</strong> {escape(report.keyword or 'n/a')}</li>",
            f"<li><strong>Generated at:</strong> {escape(report.generated_at.isoformat())}</li>",
            "</ul>",
            "<h2>Executive Summary</h2>",
            f"<p>{escape(report.executive_summary)}</p>",
            "<h2>Explanations</h2>",
        ]

        if not report.explanations:
            parts.append('<p class="empty">No explanations were generated.</p>')
        else:
            for index, explanation in enumerate(report.explanations, start=1):
                parts.extend(self._render_explanation(index, explanation))

        if report.metadata:
            parts.extend(
                [
                    "<h2>Metadata</h2>",
                    f"<pre>{escape(_format_metadata(report.metadata))}</pre>",
                ]
            )

        parts.extend(["</body>", "</html>", ""])
        return "\n".join(parts)

    def _render_explanation(self, index: int, explanation: Explanation) -> list[str]:
        parts = [
            f"<h3>Explanation {index}</h3>",
            '<ul class="meta">',
            f"<li><strong>Event ID:</strong> {escape(explanation.event_id)}</li>",
            f"<li><strong>Anomaly type:</strong> {escape(explanation.anomaly_type.value)}</li>",
            f"<li><strong>Confidence:</strong> {explanation.confidence:.2f}</li>",
            "</ul>",
            f"<p>{escape(explanation.summary)}</p>",
            "<h4>Causes</h4>",
        ]
        if not explanation.causes:
            parts.append('<p class="empty">No causes were identified.</p>')
            return parts
        for cause_index, cause in enumerate(explanation.causes, start=1):
            parts.append(self._render_cause(cause_index, cause))
        return parts

    @staticmethod
    def _render_cause(index: int, cause: Cause) -> str:
        title = escape(cause.title or cause.source)
        url = cause.url
        if url:
            href = escape(url, quote=True)
            url_html = f'<a href="{href}">{escape(url)}</a>'
        else:
            url_html = "n/a"
        lines = [
            f'<div class="cause"><p class="cause-title">{index}. {title} ({cause.score:.2f})</p>',
            f"<ul><li>Source: {escape(cause.source)}</li>",
            f"<li>URL: {url_html}</li>",
        ]
        if cause.summary:
            lines.append(f"<li>Summary: {escape(cause.summary)}</li>")
        lines.append("</ul></div>")
        return "\n".join(lines)


def _format_metadata(metadata: dict) -> str:
    return json.dumps(metadata, indent=2, default=str)
