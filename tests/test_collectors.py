from datetime import datetime, timezone
from pathlib import Path

import httpx
import pytest

from whytrend.collectors import (
    GoogleNewsCollector,
    HackerNewsCollector,
    RedditCollector,
    WikipediaCollector,
)
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
async def test_reddit_collector_parses_posts_and_comments(sample_event: Event) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.host == "www.reddit.com" and request.url.path.endswith("/access_token"):
            return httpx.Response(200, json={"access_token": "test-token", "token_type": "bearer"})

        assert request.url.host == "oauth.reddit.com"
        assert request.url.path == "/search"
        assert request.url.params["q"] == "Python"
        assert request.headers["Authorization"] == "Bearer test-token"
        return httpx.Response(
            200,
            json={
                "data": {
                    "children": [
                        {
                            "kind": "t3",
                            "data": {
                                "id": "abc123",
                                "title": "Python 3.13 discussion",
                                "selftext": "Release notes and community reactions.",
                                "permalink": "/r/Python/comments/abc123/python_313/",
                                "subreddit": "Python",
                                "score": 420,
                                "created_utc": 1767480000,
                            },
                        },
                        {
                            "kind": "t1",
                            "data": {
                                "id": "cmt1",
                                "body": "This release looks solid for async workloads.",
                                "link_title": "Python 3.13 released",
                                "permalink": "/r/Python/comments/abc123/python_313/cmt1/",
                                "subreddit": "Python",
                                "score": 12,
                                "created_utc": 1767483600,
                            },
                        },
                    ]
                }
            },
        )

    collector = RedditCollector(
        client_id="id",
        client_secret="secret",
        client=_mock_transport(handler),
    )
    evidences = await collector.collect(sample_event)

    assert collector.name == "reddit"
    assert len(evidences) == 2
    assert evidences[0].title == "Python 3.13 discussion"
    assert evidences[0].source_name == "reddit"
    assert evidences[0].url == "https://www.reddit.com/r/Python/comments/abc123/python_313/"
    assert evidences[0].metadata["subreddit"] == "Python"
    assert evidences[0].metadata["kind"] == "t3"
    assert evidences[1].title == "Python 3.13 released"
    assert evidences[1].snippet.startswith("This release looks solid")
    assert evidences[1].metadata["kind"] == "t1"


@pytest.mark.asyncio
async def test_reddit_collector_filters_outside_window(sample_event: Event) -> None:
    collector = RedditCollector(access_token="test-token")
    evidences = collector._parse_children(
        [
            {
                "kind": "t3",
                "data": {
                    "id": "in",
                    "title": "Inside window",
                    "permalink": "/r/Python/comments/in/inside/",
                    "subreddit": "Python",
                    "created_utc": 1767480000,  # 2026-01-03
                },
            },
            {
                "kind": "t3",
                "data": {
                    "id": "out",
                    "title": "Outside window",
                    "permalink": "/r/Python/comments/out/outside/",
                    "subreddit": "Python",
                    "created_utc": 1768000000,  # after window_end
                },
            },
        ],
        window_start=sample_event.window_start,
        window_end=sample_event.window_end,
    )

    assert len(evidences) == 1
    assert evidences[0].title == "Inside window"


@pytest.mark.asyncio
async def test_reddit_collector_uses_access_token_without_oauth(sample_event: Event) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.host == "oauth.reddit.com"
        assert request.headers["Authorization"] == "Bearer injected-token"
        return httpx.Response(200, json={"data": {"children": []}})

    collector = RedditCollector(
        access_token="injected-token",
        client=_mock_transport(handler),
    )
    assert await collector.collect(sample_event) == []


@pytest.mark.asyncio
async def test_reddit_collector_searches_subreddit_allowlist(sample_event: Event) -> None:
    seen_paths: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.host == "www.reddit.com":
            return httpx.Response(200, json={"access_token": "token", "token_type": "bearer"})

        seen_paths.append(request.url.path)
        assert request.url.params["restrict_sr"] == "true"
        return httpx.Response(
            200,
            json={
                "data": {
                    "children": [
                        {
                            "kind": "t3",
                            "data": {
                                "id": "xyz",
                                "title": f"Post in {request.url.path}",
                                "permalink": f"{request.url.path}/comments/xyz/post/",
                                "subreddit": request.url.path.split("/")[2],
                                "created_utc": 1767480000,
                            },
                        }
                    ]
                }
            },
        )

    collector = RedditCollector(
        client_id="id",
        client_secret="secret",
        subreddits=["Python", "MachineLearning"],
        max_results=5,
        client=_mock_transport(handler),
    )
    evidences = await collector.collect(sample_event)

    assert seen_paths == ["/r/Python/search", "/r/MachineLearning/search"]
    assert len(evidences) == 2


@pytest.mark.asyncio
async def test_reddit_collector_requires_credentials(
    sample_event: Event,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("REDDIT_CLIENT_ID", raising=False)
    monkeypatch.delenv("REDDIT_CLIENT_SECRET", raising=False)
    collector = RedditCollector(client=_mock_transport(lambda request: httpx.Response(500)))

    with pytest.raises(ValueError, match="REDDIT_CLIENT_ID"):
        await collector.collect(sample_event)


@pytest.mark.asyncio
async def test_collectors_return_empty_list_for_invalid_payload(sample_event: Event) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"unexpected": True})

    hn_collector = HackerNewsCollector(client=_mock_transport(handler))
    wiki_collector = WikipediaCollector(client=_mock_transport(handler))
    news_collector = GoogleNewsCollector(client=_mock_transport(handler))
    reddit_collector = RedditCollector(
        access_token="token",
        client=_mock_transport(handler),
    )

    assert await hn_collector.collect(sample_event) == []
    assert await wiki_collector.collect(sample_event) == []
    assert await news_collector.collect(sample_event) == []
    assert await reddit_collector.collect(sample_event) == []


def test_collector_validation() -> None:
    with pytest.raises(ValueError, match="max_results must be >= 1"):
        HackerNewsCollector(max_results=0)

    with pytest.raises(ValueError, match="language cannot be empty"):
        WikipediaCollector(language="  ")

    with pytest.raises(ValueError, match="max_results must be >= 1"):
        GoogleNewsCollector(max_results=0)

    with pytest.raises(ValueError, match="region cannot be empty"):
        GoogleNewsCollector(region="  ")

    with pytest.raises(ValueError, match="max_results must be >= 1"):
        RedditCollector(max_results=0)

    with pytest.raises(ValueError, match="user_agent cannot be empty"):
        RedditCollector(user_agent="  ")

    with pytest.raises(ValueError, match="subreddits cannot be empty"):
        RedditCollector(subreddits=["  "])
