"""Mock LLM provider for tests and offline development."""

from __future__ import annotations

from whytrend.core.models import Cause, Event, Evidence, Explanation
from whytrend.core.protocols import BaseLLMProvider
from whytrend.detectors._utils import clamp_score
from whytrend.llm._parsing import fallback_explanation


class MockLLMProvider(BaseLLMProvider):
    """Return a deterministic explanation without calling an external model."""

    def __init__(self, *, summary_prefix: str = "Likely cause") -> None:
        self._summary_prefix = summary_prefix

    @property
    def name(self) -> str:
        return "mock"

    async def explain(self, event: Event, evidences: list[Evidence]) -> Explanation:
        if not evidences:
            return fallback_explanation(event, evidences)

        top = max(evidences, key=lambda item: item.relevance_score)
        return Explanation(
            event_id=event.id,
            anomaly_type=event.anomaly_type,
            confidence=clamp_score(max(top.relevance_score, event.detection_score)),
            summary=(
                f"{self._summary_prefix}: interest in '{event.keyword}' "
                f'spiked due to "{top.title}".'
            ),
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
