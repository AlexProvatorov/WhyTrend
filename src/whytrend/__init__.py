"""WhyTrend — Explainable Time Series Analysis Framework."""

from whytrend._version import __version__
from whytrend.collectors import (
    GitHubReleasesCollector,
    GoogleNewsCollector,
    HackerNewsCollector,
    RedditCollector,
    RSSFeedCollector,
    StackOverflowCollector,
    WikipediaCollector,
)
from whytrend.detectors import ProphetDetector, RiverDetector, RupturesDetector, ZScoreDetector
from whytrend.explainers import (
    AnthropicExplainer,
    DeepSeekExplainer,
    GeminiExplainer,
    LLMExplainer,
    MockLLMProvider,
    OllamaExplainer,
    OpenAIExplainer,
)
from whytrend.pipeline import Pipeline
from whytrend.rankers import BM25Ranker, EmbeddingRanker
from whytrend.report import JSONRenderer, MarkdownRenderer
from whytrend.sources import CSVSource, GoogleTrends, PandasSource, ParquetSource

__author__ = "Alexander Provatorov"

__all__ = [
    "AnthropicExplainer",
    "BM25Ranker",
    "CSVSource",
    "DeepSeekExplainer",
    "EmbeddingRanker",
    "GeminiExplainer",
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
    "RiverDetector",
    "RupturesDetector",
    "StackOverflowCollector",
    "WikipediaCollector",
    "ZScoreDetector",
    "__version__",
]
