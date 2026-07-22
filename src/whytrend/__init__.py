"""WhyTrend — Explainable Time Series Analysis Framework."""

from whytrend.collectors import (
    GitHubReleasesCollector,
    GoogleNewsCollector,
    HackerNewsCollector,
    RSSFeedCollector,
    RedditCollector,
    StackOverflowCollector,
    WikipediaCollector,
)
from whytrend.detectors import ProphetDetector, ZScoreDetector
from whytrend.explainers import LLMExplainer, MockLLMProvider, OllamaExplainer, OpenAIExplainer
from whytrend.pipeline import Pipeline
from whytrend.rankers import BM25Ranker, EmbeddingRanker
from whytrend.report import JSONRenderer, MarkdownRenderer
from whytrend.sources import CSVSource, GoogleTrends, PandasSource, ParquetSource

__version__ = "0.2.0"
__author__ = "Alexander Provatorov"

__all__ = [
    "BM25Ranker",
    "CSVSource",
    "EmbeddingRanker",
    "GitHubReleasesCollector",
    "GoogleNewsCollector",
    "GoogleTrends",
    "HackerNewsCollector",
    "JSONRenderer",
    "LLMExplainer",
    "MarkdownRenderer",
    "MockLLMProvider",
    "OllamaExplainer",
    "OpenAIExplainer",
    "PandasSource",
    "ParquetSource",
    "Pipeline",
    "ProphetDetector",
    "RSSFeedCollector",
    "RedditCollector",
    "StackOverflowCollector",
    "WikipediaCollector",
    "ZScoreDetector",
    "__version__",
]
