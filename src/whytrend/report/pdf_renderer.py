"""PDF report renderer."""

from __future__ import annotations

from io import BytesIO
from typing import Any

from whytrend.core.models import Report
from whytrend.report.html_renderer import HTMLRenderer


class PDFRenderer:
    """Render a :class:`Report` as PDF bytes from HTML.

    Requires the optional extra: ``pip install 'whytrend[pdf]'``.
    """

    @property
    def name(self) -> str:
        return "pdf"

    def render(self, report: Report) -> bytes:
        html = HTMLRenderer().render(report)
        pisa = self._load_pisa()
        dest = BytesIO()
        result = pisa.CreatePDF(html, dest=dest, encoding="utf-8")
        if getattr(result, "err", 0):
            msg = "PDFRenderer failed to convert HTML to PDF"
            raise RuntimeError(msg)
        pdf = dest.getvalue()
        if not pdf:
            msg = "PDFRenderer produced an empty document"
            raise RuntimeError(msg)
        return pdf

    @staticmethod
    def _load_pisa() -> Any:
        try:
            from xhtml2pdf import pisa
        except ImportError as exc:
            msg = "PDFRenderer requires xhtml2pdf; install with: pip install 'whytrend[pdf]'"
            raise ImportError(msg) from exc
        return pisa
