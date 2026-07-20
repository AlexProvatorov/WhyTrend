"""Shared helpers for evidence ranking."""

from __future__ import annotations

import re

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
