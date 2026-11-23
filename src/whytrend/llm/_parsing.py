"""Parse and validate LLM JSON responses."""

from __future__ import annotations

import json
import re
from typing import Any

from pydantic import BaseModel, Field, ValidationError

from whytrend.core.enums import EvidenceVerdict
from whytrend.core.models import Cause, Event, Evidence, Explanation
from whytrend.detectors._utils import clamp_score

_LIKELY_CAUSE = re.compile(r"likely\s+cause", re.IGNORECASE)
_SENTENCE_BREAK = re.compile(r"(?<=[.!?])\s+")


class LLMResponseError(ValueError):
    """A model response that could not be turned into an explanation."""

    def __init__(self, message: str, *, raw_response: str) -> None:
        super().__init__(message)
        self.raw_response = raw_response


class LLMExplanationPayload(BaseModel):
    summary: str
    confidence: float = Field(ge=0.0, le=1.0)
    causes: list[Cause] = Field(default_factory=list)


def parse_llm_content(event: Event, content: str) -> Explanation:
    try:
        payload = json.loads(content)
    except json.JSONDecodeError as exc:
        msg = "LLM response was not valid JSON"
        raise LLMResponseError(msg, raw_response=content) from exc
    if not isinstance(payload, dict):
        msg = "LLM response JSON did not match the expected explanation schema"
        raise LLMResponseError(msg, raw_response=content)

    try:
        return explanation_from_payload(event, payload)
    except ValueError as exc:
        raise LLMResponseError(str(exc), raw_response=content) from exc


def explanation_from_payload(event: Event, payload: dict[str, Any]) -> Explanation:
    try:
        parsed = LLMExplanationPayload.model_validate(_coerce_payload(_unwrap_payload(payload)))
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
        summary = _without_likely_cause(summary)
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


def ground_causes(explanation: Explanation, evidences: list[Evidence]) -> Explanation:
    """Drop causes whose URL was not in the evidence given to the explainer.

    A summary that still describes a dropped cause is replaced. Sentences that
    name a kept cause stay; otherwise the summary is rebuilt from what remains.
    """
    allowed = {_normalize_url(item.url) for item in evidences if item.url.strip()}
    kept = [cause for cause in explanation.causes if _normalize_url(cause.url) in allowed]
    if len(kept) == len(explanation.causes):
        return explanation
    dropped = [cause for cause in explanation.causes if cause not in kept]
    return explanation.model_copy(
        update={
            "causes": kept,
            "summary": _summary_without_dropped_causes(explanation.summary, kept, dropped),
        }
    )


def _unwrap_payload(payload: dict[str, Any]) -> dict[str, Any]:
    candidate = _alias_summary(payload)
    if _is_explanation(candidate):
        return candidate
    for key in ("response_schema", "response"):
        nested = payload.get(key)
        if isinstance(nested, dict):
            nested = _alias_summary(nested)
            if _is_explanation(nested):
                return nested
    return candidate


def _alias_summary(payload: dict[str, Any]) -> dict[str, Any]:
    if "summary" in payload or not isinstance(payload.get("likely_cause"), str):
        return payload
    return {**payload, "summary": payload["likely_cause"]}


def _is_explanation(payload: dict[str, Any]) -> bool:
    return "summary" in payload and "confidence" in payload


def _coerce_payload(payload: dict[str, Any]) -> dict[str, Any]:
    data = dict(payload)
    if "confidence" in data:
        data["confidence"] = _as_float(data["confidence"])
    causes = data.get("causes")
    if isinstance(causes, list):
        data["causes"] = _usable_causes(causes)
    return data


def _usable_causes(causes: list[Any]) -> list[dict[str, Any]]:
    """Keep causes that match the schema and drop the rest.

    A model sometimes wraps a valid summary around a cause object that has
    no URL or score. That cause is not evidence; the summary still is.
    """
    usable: list[dict[str, Any]] = []
    for cause in causes:
        if not isinstance(cause, dict):
            continue
        candidate = dict(cause)
        if "score" in candidate:
            candidate["score"] = _as_float(candidate["score"])
        try:
            Cause.model_validate(candidate)
        except ValidationError:
            continue
        usable.append(candidate)
    return usable


def _as_float(value: Any) -> Any:
    if isinstance(value, str):
        try:
            return float(value.strip())
        except ValueError:
            return value
    return value


def _without_likely_cause(summary: str) -> str:
    if _LIKELY_CAUSE.search(summary) is None:
        return summary
    rewritten = re.sub(r"(?i)^likely\s+cause\b", "Correlation only", summary, count=1)
    return re.sub(r"(?i)\blikely\s+cause\b", "correlation", rewritten)


def _normalize_url(url: str) -> str:
    return url.strip().rstrip("/")


def _summary_without_dropped_causes(
    summary: str,
    kept: list[Cause],
    dropped: list[Cause],
) -> str:
    dropped_needles = _cause_needles(dropped)
    kept_needles = _cause_needles(kept)
    sentences = [part.strip() for part in _SENTENCE_BREAK.split(summary) if part.strip()]
    grounded = []
    for sentence in sentences:
        lowered = sentence.lower()
        if any(needle in lowered for needle in dropped_needles):
            continue
        if kept_needles and any(needle in lowered for needle in kept_needles):
            grounded.append(sentence)
    if grounded:
        return " ".join(grounded)
    return _summary_from_causes(kept)


def _summary_from_causes(causes: list[Cause]) -> str:
    if not causes:
        return "No cited cause was present in the collected evidence."
    names = [cause.title.strip() or cause.source for cause in causes]
    if len(names) == 1:
        return f"The cited evidence is {names[0]}."
    return f"The cited evidence is {', '.join(names[:-1])} and {names[-1]}."


def _cause_needles(causes: list[Cause]) -> list[str]:
    needles: list[str] = []
    for cause in causes:
        for value in (cause.title, cause.url):
            text = value.strip().lower().rstrip("/")
            if len(text) >= 3:
                needles.append(text)
    return needles
