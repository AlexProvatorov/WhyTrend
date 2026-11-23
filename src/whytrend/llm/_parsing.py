"""Parse and validate LLM JSON responses."""

from __future__ import annotations

import json
from typing import Any

from pydantic import BaseModel, Field, ValidationError

from whytrend.core.enums import EvidenceVerdict
from whytrend.core.models import Cause, Event, Evidence, Explanation
from whytrend.detectors._utils import clamp_score


class LLMExplanationPayload(BaseModel):
    summary: str
    confidence: float = Field(ge=0.0, le=1.0)
    causes: list[Cause] = Field(default_factory=list)


def parse_llm_content(event: Event, content: str) -> Explanation:
    try:
        payload = json.loads(content)
    except json.JSONDecodeError as exc:
        msg = "LLM response was not valid JSON"
        raise ValueError(msg) from exc

    return explanation_from_payload(event, payload)


def explanation_from_payload(event: Event, payload: dict[str, Any]) -> Explanation:
    try:
        parsed = LLMExplanationPayload.model_validate(payload)
    except ValidationError as exc:
        msg = "LLM response JSON did not match the expected explanation schema"
        raise ValueError(msg) from exc

    return Explanation(
        event_id=event.id,
        anomaly_type=event.anomaly_type,
        confidence=parsed.confidence,
        summary=parsed.summary.strip(),
        causes=parsed.causes,
        metadata={"llm_parsed": True},
    )


def evidence_insufficient_summary(event: Event) -> str:
    """Summary used when the evidence filter leaves nothing worth explaining."""
    return (
        f"Evidence is insufficient to explain the {event.anomaly_type.value} in '{event.keyword}'."
    )


def apply_verdict(
    explanation: Explanation,
    verdict: EvidenceVerdict,
    event: Event,
) -> Explanation:
    """Stamp ``verdict`` onto an explanation and keep the summary consistent with it."""
    summary = explanation.summary
    causes = list(explanation.causes)
    if verdict is EvidenceVerdict.EVIDENCE_INSUFFICIENT:
        summary = evidence_insufficient_summary(event)
        causes = []
    elif verdict is EvidenceVerdict.CORRELATION_ONLY:
        summary = summary.replace("Likely cause", "Correlation only")
    return explanation.model_copy(update={"verdict": verdict, "summary": summary, "causes": causes})


def fallback_explanation(
    event: Event,
    evidences: list[Evidence],
    *,
    verdict: EvidenceVerdict | None = None,
) -> Explanation:
    """Build a deterministic explanation when no LLM is available."""
    if verdict is EvidenceVerdict.EVIDENCE_INSUFFICIENT or not evidences:
        summary = (
            evidence_insufficient_summary(event)
            if verdict is EvidenceVerdict.EVIDENCE_INSUFFICIENT
            else (
                f"A {event.anomaly_type.value} was detected for '{event.keyword}', "
                "but no supporting external evidence was found."
            )
        )
        return Explanation(
            event_id=event.id,
            anomaly_type=event.anomaly_type,
            confidence=clamp_score(event.detection_score * 0.5),
            verdict=verdict,
            summary=summary,
            causes=[],
            metadata={"fallback": True, "evidence_count": 0},
        )

    top = max(evidences, key=lambda item: item.relevance_score)
    if verdict is EvidenceVerdict.CORRELATION_ONLY:
        summary = (
            f"The {event.anomaly_type.value} in '{event.keyword}' correlates with "
            f'"{top.title}" from {top.source_name}.'
        )
    else:
        summary = (
            f"The {event.anomaly_type.value} in '{event.keyword}' may be related to "
            f'"{top.title}" from {top.source_name}.'
        )
    return Explanation(
        event_id=event.id,
        anomaly_type=event.anomaly_type,
        confidence=clamp_score(max(top.relevance_score, event.detection_score * 0.5)),
        verdict=verdict,
        summary=summary,
        causes=[
            Cause(
                source=top.source_name,
                score=clamp_score(top.relevance_score),
                url=top.url,
                title=top.title,
                summary=top.snippet,
            )
        ],
        metadata={"fallback": True, "evidence_count": len(evidences)},
    )
