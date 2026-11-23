"""Shared helpers for evidence ranking."""

from __future__ import annotations

import re
from datetime import UTC, datetime

from whytrend.core.models import Event, Evidence
from whytrend.detectors._utils import clamp_score

_TOKEN_PATTERN = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    return _TOKEN_PATTERN.findall(text.lower())


def build_query(event: Event) -> str:
    date = event.timestamp.date().isoformat()
    return f"{event.keyword} {event.anomaly_type.value} {date}"


def evidence_text(evidence: Evidence) -> str:
    return " ".join(
        part for part in (evidence.title, evidence.snippet, evidence.source_name) if part
    )


def normalize_scores(scores: list[float]) -> list[float]:
    if not scores:
        return []

    min_score = min(scores)
    max_score = max(scores)
    if max_score <= min_score:
        return [1.0 if score > 0 else 0.0 for score in scores]
    scale = max_score - min_score
    return [(score - min_score) / scale for score in scores]


def apply_relevance_scores(
    evidences: list[Evidence],
    scores: list[float],
) -> list[Evidence]:
    return [
        evidence.model_copy(update={"relevance_score": clamp_score(score)})
        for evidence, score in zip(evidences, scores, strict=True)
    ]


def validate_time_weight(time_weight: float) -> None:
    if not 0.0 <= time_weight <= 1.0:
        msg = "time_weight must be between 0 and 1"
        raise ValueError(msg)


def apply_time_proximity(
    event: Event,
    evidences: list[Evidence],
    time_weight: float,
) -> list[Evidence]:
    """Blend relevance with distance from the event timestamp.

    ``time_weight`` of 0 returns ``evidences`` unchanged. Otherwise each score
    becomes ``relevance * ((1 - time_weight) + time_weight * time_factor)``,
    clamped to ``[0, 1]``. ``time_factor`` is 1 at ``event.timestamp``, 0 at
    either window edge, and negative outside the window. A missing
    ``published_at`` keeps factor 1.
    """
    if time_weight == 0.0:
        return evidences

    adjusted: list[Evidence] = []
    for evidence in evidences:
        factor = _time_factor(event, evidence.published_at)
        blended = evidence.relevance_score * ((1.0 - time_weight) + time_weight * factor)
        adjusted.append(evidence.model_copy(update={"relevance_score": clamp_score(blended)}))
    return adjusted


def _time_factor(event: Event, published_at: datetime | None) -> float:
    if published_at is None:
        return 1.0

    published = _as_utc(published_at)
    timestamp = _as_utc(event.timestamp)
    if published == timestamp:
        return 1.0

    if published < timestamp:
        span = (timestamp - _as_utc(event.window_start)).total_seconds()
        if span <= 0:
            return 0.0
        return (published - _as_utc(event.window_start)).total_seconds() / span

    span = (_as_utc(event.window_end) - timestamp).total_seconds()
    if span <= 0:
        return 0.0
    return (_as_utc(event.window_end) - published).total_seconds() / span


def _as_utc(moment: datetime) -> datetime:
    if moment.tzinfo is None:
        return moment.replace(tzinfo=UTC)
    return moment.astimezone(UTC)
