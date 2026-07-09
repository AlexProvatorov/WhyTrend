"""Context collectors for external evidence sources."""

from whytrend.collectors.hacker_news import HackerNewsCollector
from whytrend.collectors.wikipedia import WikipediaCollector

__all__ = [
    "HackerNewsCollector",
    "WikipediaCollector",
]
