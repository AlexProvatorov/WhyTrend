"""BM25 evidence ranker."""

from __future__ import annotations

from whytrend.core.models import Event, Evidence
from whytrend.core.protocols import BaseRanker
from whytrend.rankers._bm25 import BM25Okapi
from whytrend.rankers._utils import (
    apply_relevance_scores,
    build_query,
    evidence_text,
    normalize_scores,
    tokenize,
)


class BM25Ranker(BaseRanker):
    """Rank evidence with Okapi BM25 over title and snippet text."""

    def __init__(self, *, top_k: int | None = 10) -> None:
        if top_k is not None and top_k < 1:
            msg = "top_k must be >= 1 when provided"
            raise ValueError(msg)
        self._top_k = top_k

    @property
    def name(self) -> str:
        return "bm25"

    def rank(self, event: Event, evidences: list[Evidence]) -> list[Evidence]:
        if not evidences:
            return []

        query = tokenize(build_query(event))
        documents = [tokenize(evidence_text(evidence)) for evidence in evidences]
        if not any(documents):
            return list(evidences)

        scores = normalize_scores(BM25Okapi(documents).get_scores(query))
        ranked = sorted(
            apply_relevance_scores(evidences, scores),
            key=lambda evidence: evidence.relevance_score,
            reverse=True,
        )
        if self._top_k is None:
            return ranked
        return ranked[: self._top_k]
