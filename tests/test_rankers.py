from datetime import UTC, datetime

import numpy as np
import pytest

from tests.stubs import StubCollector, StubExplainer
from whytrend.core import AnomalyType, Evidence
from whytrend.detectors import ZScoreDetector
from whytrend.pipeline import Pipeline
from whytrend.rankers import BM25Ranker, EmbeddingRanker, HybridRanker
from whytrend.rankers._utils import apply_time_proximity, build_query
from whytrend.sources import PandasSource


@pytest.fixture
def ranking_event():
    from whytrend.core import Event

    return Event(
        anomaly_type=AnomalyType.SPIKE,
        timestamp=datetime(2026, 1, 4, tzinfo=UTC),
        window_start=datetime(2026, 1, 1, tzinfo=UTC),
        window_end=datetime(2026, 1, 7, tzinfo=UTC),
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


def _specialist_evidences() -> list[Evidence]:
    """Lexical hit, paraphrase, and an article that is only mediocre for both."""
    return [
        Evidence(
            title="Python spike exact match",
            url="https://example.com/lexical",
            snippet="Python Python spike spike",
            source_name="hacker_news",
        ),
        Evidence(
            title="mediocre Python mention",
            url="https://example.com/mediocre",
            snippet="a short note",
            source_name="wikipedia",
        ),
        Evidence(
            title="language release paraphrase",
            url="https://example.com/paraphrase",
            snippet="runtime performance",
            source_name="blog",
        ),
    ]


def _specialist_model(event):
    """Align the query with the paraphrase and push the lexical title the other way."""
    query = build_query(event)

    class _Model:
        def encode(self, sentences, normalize_embeddings=True):
            vectors = []
            for sentence in sentences:
                if sentence == query or "paraphrase" in sentence:
                    vector = (1.0, 0.0)
                elif "mediocre" in sentence:
                    vector = (0.0, 1.0)
                else:
                    vector = (-1.0, 0.0)
                vectors.append(np.array(vector, dtype=float))
            return np.vstack(vectors)

    return _Model()


def test_hybrid_ranker_lets_a_single_method_hit_beat_a_mediocre_pair(ranking_event) -> None:
    evidences = _specialist_evidences()
    ranked = HybridRanker(model=_specialist_model(ranking_event), top_k=None).rank(
        ranking_event,
        evidences,
    )

    titles = [item.title for item in ranked]
    mediocre_at = titles.index("mediocre Python mention")
    assert titles.index("Python spike exact match") < mediocre_at
    assert titles.index("language release paraphrase") < mediocre_at
    scores = [item.relevance_score for item in ranked]
    assert scores == sorted(scores, reverse=True)
    assert all(0.0 <= score <= 1.0 for score in scores)
    assert ranked[0].relevance_score > ranked[-1].relevance_score


def test_hybrid_ranker_truncates_to_top_k(ranking_event) -> None:
    ranked = HybridRanker(model=_specialist_model(ranking_event), top_k=2).rank(
        ranking_event,
        _specialist_evidences(),
    )

    assert len(ranked) == 2
    assert {item.title for item in ranked} == {
        "Python spike exact match",
        "language release paraphrase",
    }
    assert ranked[0].relevance_score >= ranked[1].relevance_score


def test_hybrid_ranker_returns_empty_for_no_evidence(ranking_event, monkeypatch) -> None:
    def fail_load(self: EmbeddingRanker) -> None:
        msg = "sentence-transformer model should not be loaded"
        raise AssertionError(msg)

    monkeypatch.setattr(EmbeddingRanker, "_load_model", fail_load)

    assert HybridRanker().rank(ranking_event, []) == []


def test_ranker_validation() -> None:
    with pytest.raises(ValueError, match="top_k must be >= 1"):
        BM25Ranker(top_k=0)

    with pytest.raises(ValueError, match="top_k must be >= 1"):
        HybridRanker(top_k=0)

    with pytest.raises(ValueError, match="model_name cannot be empty"):
        EmbeddingRanker(model_name="  ")

    with pytest.raises(ValueError, match="model_name cannot be empty"):
        HybridRanker(model_name="  ")

    with pytest.raises(ValueError, match="time_weight must be between 0 and 1"):
        BM25Ranker(time_weight=-0.1)

    with pytest.raises(ValueError, match="time_weight must be between 0 and 1"):
        EmbeddingRanker(time_weight=1.1)

    with pytest.raises(ValueError, match="time_weight must be between 0 and 1"):
        HybridRanker(time_weight=1.5)


def _same_text_pair(event) -> list[Evidence]:
    """Edge item first, so a stable text-only sort would keep it ahead."""
    shared = {
        "title": "Python release notes",
        "snippet": "Python release notes",
        "source_name": "hacker_news",
    }
    return [
        Evidence(
            url="https://example.com/edge",
            published_at=event.window_start,
            **shared,
        ),
        Evidence(
            url="https://example.com/near",
            published_at=event.timestamp,
            **shared,
        ),
    ]


class _FlatEmbeddingModel:
    def encode(self, sentences, normalize_embeddings=True):
        return np.vstack([np.array([1.0, 0.0], dtype=float) for _ in sentences])


def _scored(url: str, published_at: datetime | None, score: float = 1.0) -> Evidence:
    return Evidence(
        title="Python release notes",
        url=url,
        snippet="Python release notes",
        source_name="hacker_news",
        published_at=published_at,
        relevance_score=score,
    )


def test_time_weight_zero_leaves_scores_and_order_unchanged(ranking_event) -> None:
    evidences = _same_text_pair(ranking_event)
    baseline = BM25Ranker().rank(ranking_event, evidences)
    unchanged = BM25Ranker(time_weight=0).rank(ranking_event, evidences)

    assert [item.url for item in unchanged] == [item.url for item in baseline]
    assert [item.relevance_score for item in unchanged] == [
        item.relevance_score for item in baseline
    ]

    model = _FlatEmbeddingModel()
    base_hybrid = HybridRanker(model=model).rank(ranking_event, evidences)
    zero_hybrid = HybridRanker(model=model, time_weight=0).rank(ranking_event, evidences)
    assert [item.url for item in zero_hybrid] == [item.url for item in base_hybrid]
    assert [item.relevance_score for item in zero_hybrid] == [
        item.relevance_score for item in base_hybrid
    ]


def test_time_factor_is_one_at_the_event_and_zero_at_the_edges(ranking_event) -> None:
    midpoint = datetime(2026, 1, 2, 12, tzinfo=UTC)
    outside = datetime(2025, 12, 29, tzinfo=UTC)
    naive_event_time = datetime(2026, 1, 4)
    items = [
        _scored("https://example.com/event", ranking_event.timestamp),
        _scored("https://example.com/start", ranking_event.window_start),
        _scored("https://example.com/end", ranking_event.window_end),
        _scored("https://example.com/undated", None, score=0.4),
        _scored("https://example.com/mid", midpoint),
        _scored("https://example.com/outside", outside),
        _scored("https://example.com/naive", naive_event_time),
    ]

    unchanged = apply_time_proximity(ranking_event, items, 0)
    assert unchanged is items

    adjusted = apply_time_proximity(ranking_event, items, 1)
    scores = [item.relevance_score for item in adjusted]
    assert scores == pytest.approx([1.0, 0.0, 0.0, 0.4, 0.5, 0.0, 1.0])
    assert all(0.0 <= score <= 1.0 for score in scores)

    half = apply_time_proximity(ranking_event, items[:2], 0.5)
    assert half[0].relevance_score == pytest.approx(1.0)
    assert half[1].relevance_score == pytest.approx(0.5)


def test_nearer_publication_outranks_the_window_edge(ranking_event) -> None:
    evidences = _same_text_pair(ranking_event)

    for ranker in (
        BM25Ranker(time_weight=1),
        EmbeddingRanker(model=_FlatEmbeddingModel(), time_weight=1),
        HybridRanker(model=_FlatEmbeddingModel(), time_weight=1, top_k=None),
    ):
        ranked = ranker.rank(ranking_event, evidences)
        assert ranked[0].url == "https://example.com/near"
        assert ranked[0].relevance_score > ranked[1].relevance_score
        assert all(0.0 <= item.relevance_score <= 1.0 for item in ranked)

    fused = HybridRanker(model=_FlatEmbeddingModel(), time_weight=1, top_k=None).rank(
        ranking_event,
        evidences,
    )
    assert fused[0].relevance_score == 1.0
    # Rank 2 on both lists. Time already changed that order; RRF does not apply it again.
    assert fused[1].relevance_score == pytest.approx(61 / 62)


def test_missing_published_at_is_not_penalised(ranking_event) -> None:
    edge, _near = _same_text_pair(ranking_event)
    undated = Evidence(
        title=edge.title,
        url="https://example.com/undated",
        snippet=edge.snippet,
        source_name=edge.source_name,
        published_at=None,
    )
    evidences = [edge, undated]
    before = {
        item.url: item.relevance_score for item in BM25Ranker().rank(ranking_event, evidences)
    }
    after = BM25Ranker(time_weight=1).rank(ranking_event, evidences)

    undated_after = next(item for item in after if item.published_at is None)
    assert undated_after.relevance_score == before[undated_after.url]
    assert undated_after.relevance_score > after[-1].relevance_score


def test_items_outside_the_window_sink_and_are_kept(ranking_event) -> None:
    outside = Evidence(
        title="Python release notes",
        url="https://example.com/outside",
        snippet="Python release notes",
        source_name="hacker_news",
        published_at=datetime(2025, 12, 29, tzinfo=UTC),
    )
    near = Evidence(
        title="Python release notes",
        url="https://example.com/near",
        snippet="Python release notes",
        source_name="hacker_news",
        published_at=ranking_event.timestamp,
    )

    ranked = BM25Ranker(time_weight=1, top_k=None).rank(ranking_event, [outside, near])

    assert [item.url for item in ranked] == [near.url, outside.url]
    assert ranked[1].relevance_score == 0.0


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
