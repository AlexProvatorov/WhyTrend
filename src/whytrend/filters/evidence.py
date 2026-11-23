"""Drop weak evidence and assign an explicit verdict."""

from __future__ import annotations

from whytrend.core.enums import EvidenceVerdict
from whytrend.core.models import Cause, Evidence, Explanation
from whytrend.core.protocols import BaseFilter


class EvidenceFilter(BaseFilter):
    """Keep evidence at or above a threshold, then judge what remains.

    Items below ``keep_threshold`` are removed. The rest are ordered by
    ``relevance_score``. One item at or above ``cause_threshold`` is a likely
    cause. Anything kept below that is only a correlation. An empty result is
    insufficient evidence. After the explainer returns, confidence and cause
    scores are clamped to the ceiling for that verdict.
    """

    def __init__(
        self,
        *,
        keep_threshold: float = 0.2,
        cause_threshold: float = 0.6,
        likely_cause_cap: float = 1.0,
        correlation_only_cap: float = 0.6,
        evidence_insufficient_cap: float = 0.3,
    ) -> None:
        _check_unit_interval("keep_threshold", keep_threshold)
        _check_unit_interval("cause_threshold", cause_threshold)
        _check_unit_interval("likely_cause_cap", likely_cause_cap)
        _check_unit_interval("correlation_only_cap", correlation_only_cap)
        _check_unit_interval("evidence_insufficient_cap", evidence_insufficient_cap)
        if cause_threshold < keep_threshold:
            msg = "cause_threshold must be >= keep_threshold"
            raise ValueError(msg)
        self._keep_threshold = keep_threshold
        self._cause_threshold = cause_threshold
        self._likely_cause_cap = likely_cause_cap
        self._correlation_only_cap = correlation_only_cap
        self._evidence_insufficient_cap = evidence_insufficient_cap
        self._caps = {
            EvidenceVerdict.LIKELY_CAUSE: likely_cause_cap,
            EvidenceVerdict.CORRELATION_ONLY: correlation_only_cap,
            EvidenceVerdict.EVIDENCE_INSUFFICIENT: evidence_insufficient_cap,
        }

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

    def cap_confidence(self, explanation: Explanation) -> Explanation:
        """Clamp confidence and cause scores to the ceiling for the stamped verdict.

        A missing verdict is left alone. Values already within the ceiling are
        not rewritten. When a number is lowered, metadata records
        ``confidence_capped``.
        """
        if explanation.verdict is None:
            return explanation

        ceiling = self._caps[explanation.verdict]
        confidence = min(explanation.confidence, ceiling)
        causes = [_cap_cause(cause, ceiling) for cause in explanation.causes]
        lowered = confidence < explanation.confidence or any(
            updated.score < original.score
            for updated, original in zip(causes, explanation.causes, strict=True)
        )
        if not lowered:
            return explanation
        return explanation.model_copy(
            update={
                "confidence": confidence,
                "causes": causes,
                "metadata": {**explanation.metadata, "confidence_capped": True},
            }
        )


def _check_unit_interval(name: str, value: float) -> None:
    if not 0.0 <= value <= 1.0:
        msg = f"{name} must be between 0 and 1"
        raise ValueError(msg)


def _cap_cause(cause: Cause, ceiling: float) -> Cause:
    if cause.score <= ceiling:
        return cause
    return cause.model_copy(update={"score": ceiling})
