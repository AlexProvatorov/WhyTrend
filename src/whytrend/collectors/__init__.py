"""Context collectors for external evidence sources."""

from whytrend.collectors.github_releases import GitHubReleasesCollector
from whytrend.collectors.google_news import GoogleNewsCollector
from whytrend.collectors.hacker_news import HackerNewsCollector
from whytrend.collectors.reddit import RedditCollector
from whytrend.collectors.wikipedia import WikipediaCollector

__all__ = [
    "GitHubReleasesCollector",
    "GoogleNewsCollector",
    "HackerNewsCollector",
    "RedditCollector",
    "WikipediaCollector",
]
