"""Mock LLM provider for tests and offline development."""

from __future__ import annotations

from whytrend.core.enums import EvidenceVerdict
from whytrend.core.models import Cause, Event, Evidence, Explanation
from whytrend.core.protocols import BaseLLMProvider
from whytrend.detectors._utils import clamp_score
from whytrend.llm._parsing import apply_verdict, fallback_explanation


class MockLLMProvider(BaseLLMProvider):
    """Return a deterministic explanation without calling an external model."""

    def __init__(self, *, summary_prefix: str = "Likely cause") -> None:
        self._summary_prefix = summary_prefix

    @property
    def name(self) -> str:
        return "mock"

    async def explain(
        self,
        event: Event,
        evidences: list[Evidence],
        *,
        verdict: EvidenceVerdict | None = None,
    ) -> Explanation:
        if verdict is EvidenceVerdict.EVIDENCE_INSUFFICIENT or not evidences:
            explanation = fallback_explanation(event, evidences, verdict=verdict)
            if verdict is None:
                return explanation
            return apply_verdict(explanation, verdict, event)

        top = max(evidences, key=lambda item: item.relevance_score)
        prefix = (
            "Correlation only"
            if verdict is EvidenceVerdict.CORRELATION_ONLY
            else self._summary_prefix
        )
        explanation = Explanation(
            event_id=event.id,
            anomaly_type=event.anomaly_type,
            confidence=clamp_score(max(top.relevance_score, event.detection_score)),
            verdict=verdict,
            summary=(f"{prefix}: interest in '{event.keyword}' spiked due to \"{top.title}\"."),
            causes=[
                Cause(
                    source=top.source_name,
                    score=clamp_score(top.relevance_score),
                    url=top.url,
                    title=top.title,
                    summary=top.snippet,
                )
            ],
            metadata={"provider": self.name, "evidence_count": len(evidences)},
        )
        if verdict is None:
            return explanation
        return apply_verdict(explanation, verdict, event)
