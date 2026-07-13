from datetime import datetime, timezone
from pathlib import Path

import httpx
import pytest

from whytrend.collectors import GoogleNewsCollector, HackerNewsCollector, WikipediaCollector
from whytrend.core import AnomalyType, Event

FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"
GOOGLE_NEWS_SAMPLE_RSS = (FIXTURES_DIR / "google_news_sample.xml").read_text(encoding="utf-8")


@pytest.fixture
def sample_event() -> Event:
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


def _mock_transport(handler) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


@pytest.mark.asyncio
async def test_google_news_collector_parses_rss(sample_event: Event) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.host == "news.google.com"
        assert request.url.params["q"] == "Python"
        return httpx.Response(200, text=GOOGLE_NEWS_SAMPLE_RSS)

    collector = GoogleNewsCollector(client=_mock_transport(handler))
    evidences = await collector.collect(sample_event)

    assert len(evidences) == 1
    assert evidences[0].title == "Python 3.13 tops developer headlines"
    assert evidences[0].source_name == "google_news"
    assert evidences[0].url == "https://example.com/python-headlines"
    assert evidences[0].snippet == "Major release drives search interest."
    assert evidences[0].published_at == datetime(2026, 1, 3, 12, 0, tzinfo=timezone.utc)
    assert evidences[0].metadata["publisher"] == "Example News"


@pytest.mark.asyncio
async def test_google_news_collector_filters_articles_outside_window(sample_event: Event) -> None:
    collector = GoogleNewsCollector()
    evidences = collector._parse_rss(
        GOOGLE_NEWS_SAMPLE_RSS,
        window_start=sample_event.window_start,
        window_end=sample_event.window_end,
    )

    assert len(evidences) == 1
    assert evidences[0].title == "Python 3.13 tops developer headlines"


@pytest.mark.asyncio
async def test_google_news_collector_returns_empty_for_invalid_xml(sample_event: Event) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text="not xml")

    collector = GoogleNewsCollector(client=_mock_transport(handler))
    assert await collector.collect(sample_event) == []


@pytest.mark.asyncio
async def test_hacker_news_collector_parses_hits(sample_event: Event) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.host == "hn.algolia.com"
        assert request.url.params["query"] == "Python"
        return httpx.Response(
            200,
            json={
                "hits": [
                    {
                        "title": "Python 3.13 released",
                        "url": "https://example.com/python-3-13",
                        "story_text": "Major release with performance improvements.",
                        "created_at_i": 1767480000,
                        "objectID": "123",
                    }
                ]
            },
        )

    collector = HackerNewsCollector(client=_mock_transport(handler))
    evidences = await collector.collect(sample_event)

    assert len(evidences) == 1
    assert evidences[0].title == "Python 3.13 released"
    assert evidences[0].source_name == "hacker_news"
    assert evidences[0].published_at is not None


@pytest.mark.asyncio
async def test_hacker_news_collector_builds_fallback_item_url(sample_event: Event) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "hits": [
                    {
                        "title": "Show HN: WhyTrend",
                        "story_id": 424242,
                        "created_at_i": 1767480000,
                    }
                ]
            },
        )

    collector = HackerNewsCollector(client=_mock_transport(handler))
    evidences = await collector.collect(sample_event)

    assert evidences[0].url == "https://news.ycombinator.com/item?id=424242"


@pytest.mark.asyncio
async def test_wikipedia_collector_parses_search_results(sample_event: Event) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.host == "en.wikipedia.org"
        assert request.url.params["srsearch"] == "Python"
        return httpx.Response(
            200,
            json={
                "query": {
                    "search": [
                        {
                            "title": "Python (programming language)",
                            "snippet": "Python is a high-level programming language.",
                            "pageid": 23862,
                        }
                    ]
                }
            },
        )

    collector = WikipediaCollector(client=_mock_transport(handler))
    evidences = await collector.collect(sample_event)

    assert len(evidences) == 1
    assert evidences[0].title == "Python (programming language)"
    assert evidences[0].source_name == "wikipedia"
    assert evidences[0].url == "https://en.wikipedia.org/wiki/Python_%28programming_language%29"


@pytest.mark.asyncio
async def test_collectors_return_empty_list_for_invalid_payload(sample_event: Event) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"unexpected": True})

    hn_collector = HackerNewsCollector(client=_mock_transport(handler))
    wiki_collector = WikipediaCollector(client=_mock_transport(handler))
    news_collector = GoogleNewsCollector(client=_mock_transport(handler))

    assert await hn_collector.collect(sample_event) == []
    assert await wiki_collector.collect(sample_event) == []
    assert await news_collector.collect(sample_event) == []


def test_collector_validation() -> None:
    with pytest.raises(ValueError, match="max_results must be >= 1"):
        HackerNewsCollector(max_results=0)

    with pytest.raises(ValueError, match="language cannot be empty"):
        WikipediaCollector(language="  ")

    with pytest.raises(ValueError, match="max_results must be >= 1"):
        GoogleNewsCollector(max_results=0)

    with pytest.raises(ValueError, match="region cannot be empty"):
        GoogleNewsCollector(region="  ")
