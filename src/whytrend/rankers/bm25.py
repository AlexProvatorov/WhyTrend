"""BM25 evidence ranker."""

from __future__ import annotations

from whytrend.core.models import Event, Evidence
from whytrend.core.protocols import BaseRanker
from whytrend.rankers._bm25 import BM25Okapi
from whytrend.rankers._utils import (
    apply_relevance_scores,
    apply_time_proximity,
    build_query,
    evidence_text,
    tokenize,
    validate_time_weight,
)


class BM25Ranker(BaseRanker):
    """Rank evidence with Okapi BM25 over title and snippet text."""

    def __init__(self, *, top_k: int | None = 10, time_weight: float = 0.0) -> None:
        if top_k is not None and top_k < 1:
            msg = "top_k must be >= 1 when provided"
            raise ValueError(msg)
        validate_time_weight(time_weight)
        self._top_k = top_k
        self._time_weight = time_weight

    @property
    def name(self) -> str:
        return "bm25"

    def rank(self, event: Event, evidences: list[Evidence]) -> list[Evidence]:
        if not evidences:
            return []

        query = tokenize(build_query(event))
        documents = [tokenize(evidence_text(evidence)) for evidence in evidences]
        if not any(documents):
            return apply_time_proximity(event, list(evidences), self._time_weight)

        model = BM25Okapi(documents)
        raw_scores = model.get_scores(query)
        ceiling = model.query_ceiling(query)
        if ceiling <= 0:
            scores = [0.0] * len(raw_scores)
        else:
            scores = [score / ceiling for score in raw_scores]
        ranked = sorted(
            apply_time_proximity(
                event,
                apply_relevance_scores(evidences, scores),
                self._time_weight,
            ),
            key=lambda evidence: evidence.relevance_score,
            reverse=True,
        )
        if self._top_k is None:
            return ranked
        return ranked[: self._top_k]
