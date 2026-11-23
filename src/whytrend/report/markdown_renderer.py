"""Markdown report renderer."""

from __future__ import annotations

from whytrend.core.models import Cause, Explanation, Report
from whytrend.core.protocols import BaseReportRenderer


class MarkdownRenderer(BaseReportRenderer):
    """Render a :class:`Report` as Markdown."""

    @property
    def name(self) -> str:
        return "markdown"

    def render(self, report: Report) -> str:
        lines = [
            "# WhyTrend Report",
            "",
            f"- **Series:** {report.series_name or 'n/a'}",
            f"- **Keyword:** {report.keyword or 'n/a'}",
            f"- **Generated at:** {report.generated_at.isoformat()}",
            "",
            "## Executive Summary",
            "",
            report.executive_summary,
            "",
        ]

        if not report.explanations:
            lines.extend(["## Explanations", "", "_No explanations were generated._", ""])
        else:
            lines.append("## Explanations")
            lines.append("")
            for index, explanation in enumerate(report.explanations, start=1):
                lines.extend(self._render_explanation(index, explanation))

        if report.metadata:
            lines.extend(
                ["## Metadata", "", "```json", _format_metadata(report.metadata), "```", ""]
            )

        return "\n".join(lines).rstrip() + "\n"

    def _render_explanation(self, index: int, explanation: Explanation) -> list[str]:
        lines = [
            f"### Explanation {index}",
            "",
            f"- **Event ID:** {explanation.event_id}",
            f"- **Anomaly type:** {explanation.anomaly_type.value}",
            f"- **Confidence:** {explanation.confidence:.2f}",
        ]
        lines.extend(_event_context_lines(explanation))
        lines.extend(_merged_detection_lines(explanation))
        if explanation.verdict is not None:
            lines.append(f"- **Verdict:** {explanation.verdict.value}")
        lines.extend(["", explanation.summary, ""])

        if explanation.causes:
            lines.append("#### Causes")
            lines.append("")
            for cause_index, cause in enumerate(explanation.causes, start=1):
                lines.extend(self._render_cause(cause_index, cause))
        else:
            lines.extend(["#### Causes", "", "_No causes were identified._", ""])

        return lines

    @staticmethod
    def _render_cause(index: int, cause: Cause) -> list[str]:
        lines = [
            f"{index}. **{cause.title or cause.source}** ({cause.score:.2f})",
            f"   - Source: {cause.source}",
            f"   - URL: {cause.url}",
        ]
        if cause.summary:
            lines.append(f"   - Summary: {cause.summary}")
        lines.append("")
        return lines


def _event_context_lines(explanation: Explanation) -> list[str]:
    lines: list[str] = []
    if explanation.timestamp is not None:
        lines.append(f"- **Timestamp:** {explanation.timestamp.isoformat()}")
    if explanation.value is not None:
        lines.append(f"- **Value:** {explanation.value:g}")
    if explanation.window_start is not None and explanation.window_end is not None:
        start = explanation.window_start.isoformat()
        end = explanation.window_end.isoformat()
        lines.append(f"- **Window:** {start} to {end}")
    return lines


def _merged_detection_lines(explanation: Explanation) -> list[str]:
    merged = explanation.metadata.get("merged_detections")
    if not isinstance(merged, int) or merged <= 1:
        return []
    return [f"- **Merged detections:** {merged}"]


def _format_metadata(metadata: dict) -> str:
    import json

    return json.dumps(metadata, indent=2, default=str)
