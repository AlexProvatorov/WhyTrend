"""Embedding-based evidence ranker."""

from __future__ import annotations

from typing import Any, Protocol

import numpy as np

from whytrend.core.models import Event, Evidence
from whytrend.core.protocols import BaseRanker
from whytrend.detectors._utils import clamp_score
from whytrend.rankers._utils import (
    apply_time_proximity,
    build_query,
    evidence_text,
    validate_time_weight,
)


class _EmbeddingModel(Protocol):
    def encode(self, sentences: list[str], **kwargs: Any) -> np.ndarray: ...


class EmbeddingRanker(BaseRanker):
    """Rank evidence with sentence-transformer cosine similarity."""

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
        if not model_name.strip():
            msg = "model_name cannot be empty"
            raise ValueError(msg)
        validate_time_weight(time_weight)

        self._model_name = model_name
        self._top_k = top_k
        self._model = model
        self._time_weight = time_weight

    @property
    def name(self) -> str:
        return "embedding"

    def rank(self, event: Event, evidences: list[Evidence]) -> list[Evidence]:
        if not evidences:
            return []

        model = self._model or self._load_model()
        query = build_query(event)
        documents = [evidence_text(evidence) for evidence in evidences]

        query_vector = model.encode([query], normalize_embeddings=True)[0]
        document_vectors = model.encode(documents, normalize_embeddings=True)
        # Cosine is already in [-1, 1]. Shifting it with (cos + 1) / 2 scores an
        # unrelated page near 0.5 and lets a passing mention clear cause_threshold.
        scores = [clamp_score(float(np.dot(query_vector, vector))) for vector in document_vectors]

        ranked = sorted(
            apply_time_proximity(
                event,
                [
                    evidence.model_copy(update={"relevance_score": score})
                    for evidence, score in zip(evidences, scores, strict=True)
                ],
                self._time_weight,
            ),
            key=lambda evidence: evidence.relevance_score,
            reverse=True,
        )
        if self._top_k is None:
            return ranked
        return ranked[: self._top_k]

    def _load_model(self) -> _EmbeddingModel:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            msg = (
                "EmbeddingRanker requires sentence-transformers; "
                "install with: pip install 'whytrend[ranking]'"
            )
            raise ImportError(msg) from exc

        self._model = SentenceTransformer(self._model_name)
        return self._model
