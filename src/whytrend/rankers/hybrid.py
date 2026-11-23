"""Hybrid BM25 and embedding ranker."""

from __future__ import annotations

from whytrend.core.models import Event, Evidence
from whytrend.core.protocols import BaseRanker
from whytrend.detectors._utils import clamp_score
from whytrend.rankers._utils import validate_time_weight
from whytrend.rankers.bm25 import BM25Ranker
from whytrend.rankers.embedding import EmbeddingRanker, _EmbeddingModel

# Standard RRF constant from Cormack, Clarke, and Buettcher (2009).
_RRF_K = 60


class HybridRanker(BaseRanker):
    """Fuse BM25 and embedding rankings with reciprocal rank fusion.

    Both rankers score the full evidence list and apply ``time_weight``
    before fusion, so RRF sees that order. RRF only decides order and
    ``top_k``. ``relevance_score`` is the mean of the two ranker scores, so a
    weak list is not stretched up to ``1.0``.
    """

    def __init__(
        self,
        *,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        top_k: int | None = 10,
        model: _EmbeddingModel | None = None,
        time_weight: float = 0.0,
    ) -> None:
        if top_k is not None and top_k < 1:
            msg = "top_k must be >= 1 when provided"
            raise ValueError(msg)
        validate_time_weight(time_weight)
        self._top_k = top_k
        self._time_weight = time_weight
        # Full lists: truncating before fusion would drop a hit that only one method ranks highly.
        self._bm25 = BM25Ranker(top_k=None, time_weight=time_weight)
        self._embedding = EmbeddingRanker(
            model_name=model_name,
            top_k=None,
            model=model,
            time_weight=time_weight,
        )

    @property
    def name(self) -> str:
        return "hybrid"

    def rank(self, event: Event, evidences: list[Evidence]) -> list[Evidence]:
        if not evidences:
            return []

        bm25_ranked = self._bm25.rank(event, evidences)
        try:
            embedding_ranked = self._embedding.rank(event, evidences)
        except ImportError as exc:
            if "whytrend[ranking]" not in str(exc):
                raise
            msg = (
                "HybridRanker requires sentence-transformers; "
                "install with: pip install 'whytrend[ranking]'"
            )
            raise ImportError(msg) from exc

        rrf_scores = _rrf_scores(
            [
                _positions(evidences, bm25_ranked),
                _positions(evidences, embedding_ranked),
            ],
            k=_RRF_K,
        )
        fused = []
        for index, evidence in enumerate(evidences):
            absolute = (
                _score_of(evidence, bm25_ranked) + _score_of(evidence, embedding_ranked)
            ) / 2
            fused.append((rrf_scores[index], clamp_score(absolute), evidence))
        fused.sort(key=lambda item: item[0], reverse=True)
        ranked = [
            evidence.model_copy(update={"relevance_score": score})
            for _rrf, score, evidence in fused
        ]
        if self._top_k is None:
            return ranked
        return ranked[: self._top_k]


def _score_of(evidence: Evidence, ranked: list[Evidence]) -> float:
    for item in ranked:
        if _same_evidence(evidence, item):
            return item.relevance_score
    return 0.0


def _positions(original: list[Evidence], ranked: list[Evidence]) -> list[int]:
    """Return a 1-based rank for each item in ``original``.

    Missing items, which these rankers do not produce when ``top_k`` is unset,
    receive a rank one past the end of ``ranked``.
    """
    positions = [len(ranked) + 1] * len(original)
    used = [False] * len(original)
    for rank, item in enumerate(ranked, start=1):
        for index, source in enumerate(original):
            if used[index]:
                continue
            if _same_evidence(source, item):
                used[index] = True
                positions[index] = rank
                break
    return positions


def _same_evidence(left: Evidence, right: Evidence) -> bool:
    left_data = left.model_dump()
    right_data = right.model_dump()
    left_data.pop("relevance_score", None)
    right_data.pop("relevance_score", None)
    return left_data == right_data


def _rrf_scores(rank_lists: list[list[int]], *, k: int) -> list[float]:
    """Scale fused reciprocal ranks so first place on every list is ``1.0``."""
    method_count = len(rank_lists)
    scale = (k + 1) / method_count
    width = len(rank_lists[0])
    return [scale * sum(1.0 / (k + ranks[index]) for ranks in rank_lists) for index in range(width)]
