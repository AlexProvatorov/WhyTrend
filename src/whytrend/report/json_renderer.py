"""JSON report renderer."""

from __future__ import annotations

from whytrend.core.models import Report
from whytrend.core.protocols import BaseReportRenderer


class JSONRenderer(BaseReportRenderer):
    """Render a :class:`Report` as formatted JSON."""

    def __init__(self, *, indent: int = 2) -> None:
        if indent < 0:
            msg = "indent must be >= 0"
            raise ValueError(msg)
        self._indent = indent

    @property
    def name(self) -> str:
        return "json"

    def render(self, report: Report) -> str:
        if self._indent == 0:
            return report.model_dump_json()
        return report.model_dump_json(indent=self._indent)
