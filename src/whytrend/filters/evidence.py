"""Drop weak evidence and assign an explicit verdict."""

from __future__ import annotations

from whytrend.core.enums import EvidenceVerdict
from whytrend.core.models import Evidence
from whytrend.core.protocols import BaseFilter


class EvidenceFilter(BaseFilter):
    """Keep evidence at or above a threshold, then judge what remains.

    Items below ``keep_threshold`` are removed. The rest are ordered by
    ``relevance_score``. One item at or above ``cause_threshold`` is a likely
    cause. Anything kept below that is only a correlation. An empty result is
    insufficient evidence.
    """

    def __init__(self, *, keep_threshold: float = 0.2, cause_threshold: float = 0.6) -> None:
        if not 0.0 <= keep_threshold <= 1.0:
            msg = "keep_threshold must be between 0 and 1"
            raise ValueError(msg)
        if not 0.0 <= cause_threshold <= 1.0:
            msg = "cause_threshold must be between 0 and 1"
            raise ValueError(msg)
        if cause_threshold < keep_threshold:
            msg = "cause_threshold must be >= keep_threshold"
            raise ValueError(msg)
        self._keep_threshold = keep_threshold
        self._cause_threshold = cause_threshold

    @property
    def name(self) -> str:
        return "evidence"

    def filter(self, evidences: list[Evidence]) -> tuple[list[Evidence], EvidenceVerdict]:
        kept = [
            evidence for evidence in evidences if evidence.relevance_score >= self._keep_threshold
        ]
        ranked = sorted(kept, key=lambda evidence: evidence.relevance_score, reverse=True)
        if not ranked:
            return [], EvidenceVerdict.EVIDENCE_INSUFFICIENT
        if ranked[0].relevance_score >= self._cause_threshold:
            return ranked, EvidenceVerdict.LIKELY_CAUSE
        return ranked, EvidenceVerdict.CORRELATION_ONLY
