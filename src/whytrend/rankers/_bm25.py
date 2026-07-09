"""Minimal Okapi BM25 implementation."""

from __future__ import annotations

import math
from collections import Counter


class BM25Okapi:
    """Rank documents against a query using BM25."""

    def __init__(
        self,
        corpus: list[list[str]],
        *,
        k1: float = 1.5,
        b: float = 0.75,
    ) -> None:
        if not corpus:
            msg = "corpus cannot be empty"
            raise ValueError(msg)

        self._k1 = k1
        self._b = b
        self._corpus = corpus
        self._doc_count = len(corpus)
        self._doc_lengths = [len(document) for document in corpus]
        self._avg_doc_length = sum(self._doc_lengths) / self._doc_count
        self._doc_freqs: list[Counter[str]] = [Counter(document) for document in corpus]
        self._idf = self._compute_idf()

    def _compute_idf(self) -> dict[str, float]:
        document_frequency: Counter[str] = Counter()
        for document in self._corpus:
            document_frequency.update(set(document))

        idf: dict[str, float] = {}
        for term, freq in document_frequency.items():
            idf[term] = math.log(1 + (self._doc_count - freq + 0.5) / (freq + 0.5))
        return idf

    def get_scores(self, query: list[str]) -> list[float]:
        if not query:
            return [0.0] * self._doc_count

        scores = [0.0] * self._doc_count
        term_counts = Counter(query)
        for term, query_freq in term_counts.items():
            if term not in self._idf:
                continue

            idf = self._idf[term]
            for index, doc_freq in enumerate(self._doc_freqs):
                if term not in doc_freq:
                    continue

                term_freq = doc_freq[term]
                doc_length = self._doc_lengths[index]
                numerator = term_freq * (self._k1 + 1)
                denominator = term_freq + self._k1 * (
                    1 - self._b + self._b * doc_length / self._avg_doc_length
                )
                scores[index] += idf * (numerator / denominator) * query_freq

        return scores
