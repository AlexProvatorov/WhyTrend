from datetime import datetime, timezone

import numpy as np
import pytest

from whytrend.core import AnomalyType, Evidence
from whytrend.pipeline import Pipeline
from whytrend.rankers import BM25Ranker, EmbeddingRanker
from whytrend.sources import PandasSource
from tests.stubs import StubCollector, StubExplainer
from whytrend.detectors import ZScoreDetector


@pytest.fixture
def ranking_event():
    from whytrend.core import Event

    return Event(
        anomaly_type=AnomalyType.SPIKE,
        timestamp=datetime(2026, 1, 4, tzinfo=timezone.utc),
        window_start=datetime(2026, 1, 1, tzinfo=timezone.utc),
        window_end=datetime(2026, 1, 7, tzinfo=timezone.utc),
        keyword="Python",
        series_name="python_interest",
        value=610.0,
        detection_score=0.95,
    )


@pytest.fixture
def sample_evidences() -> list[Evidence]:
    return [
        Evidence(
            title="Python 3.13 released",
            url="https://example.com/python-3-13",
            snippet="Major Python release with performance improvements.",
            source_name="hacker_news",
        ),
        Evidence(
            title="Unrelated football match results",
            url="https://example.com/football",
            snippet="Weekend sports roundup.",
            source_name="wikipedia",
        ),
    ]


def test_bm25_ranker_orders_relevant_evidence_first(ranking_event, sample_evidences) -> None:
    ranked = BM25Ranker(top_k=2).rank(ranking_event, sample_evidences)

    assert len(ranked) == 2
    assert ranked[0].title == "Python 3.13 released"
    assert ranked[0].relevance_score >= ranked[1].relevance_score
    assert ranked[0].relevance_score > 0.0


def test_bm25_ranker_returns_empty_for_no_evidence(ranking_event) -> None:
    assert BM25Ranker().rank(ranking_event, []) == []


def test_embedding_ranker_orders_relevant_evidence_first(ranking_event, sample_evidences) -> None:
    class FakeEmbeddingModel:
        def encode(self, sentences, normalize_embeddings=True):
            vectors = []
            for sentence in sentences:
                if "Python" in sentence:
                    vectors.append(np.array([1.0, 0.0], dtype=float))
                else:
                    vectors.append(np.array([0.0, 1.0], dtype=float))
            return np.vstack(vectors)

    ranked = EmbeddingRanker(model=FakeEmbeddingModel(), top_k=2).rank(
        ranking_event,
        sample_evidences,
    )

    assert ranked[0].title == "Python 3.13 released"
    assert ranked[0].relevance_score == 1.0
    assert ranked[1].relevance_score == 0.5


def test_ranker_validation() -> None:
    with pytest.raises(ValueError, match="top_k must be >= 1"):
        BM25Ranker(top_k=0)

    with pytest.raises(ValueError, match="model_name cannot be empty"):
        EmbeddingRanker(model_name="  ")


@pytest.mark.asyncio
async def test_pipeline_applies_ranker_before_explainer(
    spike_dataframe,
    sample_evidences,
) -> None:
    class RecordingExplainer(StubExplainer):
        def __init__(self) -> None:
            super().__init__(name="recording-explainer")
            self.last_evidences: list[Evidence] = []

        async def explain(self, event, evidences):
            self.last_evidences = list(evidences)
            return await super().explain(event, evidences)

    explainer = RecordingExplainer()
    pipeline = (
        Pipeline(window_days=2)
        .add_source(
            PandasSource(
                spike_dataframe,
                name="python_interest",
                keyword="Python",
            )
        )
        .add_detector(ZScoreDetector(threshold=1.0))
        .add_collector(StubCollector(sample_evidences))
        .add_ranker(BM25Ranker(top_k=1))
        .add_explainer(explainer)
    )

    report = await pipeline.arun()

    assert report.metadata["ranker"] == "bm25"
    assert len(explainer.last_evidences) == 1
    assert explainer.last_evidences[0].title == "Python 3.13 released"
    assert explainer.last_evidences[0].relevance_score > 0.0
