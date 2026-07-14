"""Context collectors for external evidence sources."""

from whytrend.collectors.google_news import GoogleNewsCollector
from whytrend.collectors.hacker_news import HackerNewsCollector
from whytrend.collectors.reddit import RedditCollector
from whytrend.collectors.wikipedia import WikipediaCollector

__all__ = [
    "GoogleNewsCollector",
    "HackerNewsCollector",
    "RedditCollector",
    "WikipediaCollector",
]
