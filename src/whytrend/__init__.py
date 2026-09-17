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
    AzureOpenAIExplainer,
    DeepSeekExplainer,
    GeminiExplainer,
    LLMExplainer,
    MockLLMProvider,
    OllamaExplainer,
    OpenAIExplainer,
    OpenRouterExplainer,
)
from whytrend.pipeline import Pipeline
from whytrend.plugins import PluginError, UnknownPluginError, list_plugins, load_plugin
from whytrend.rankers import BM25Ranker, EmbeddingRanker
from whytrend.report import HTMLRenderer, JSONRenderer, MarkdownRenderer, PDFRenderer
from whytrend.sources import CSVSource, GoogleTrends, PandasSource, ParquetSource

__author__ = "Alexander Provatorov"

__all__ = [
    "AnthropicExplainer",
    "AzureOpenAIExplainer",
    "BM25Ranker",
    "CSVSource",
    "DeepSeekExplainer",
    "EmbeddingRanker",
    "GeminiExplainer",
    "GitHubReleasesCollector",
    "GoogleNewsCollector",
    "GoogleTrends",
    "HackerNewsCollector",
    "HTMLRenderer",
    "JSONRenderer",
    "LLMExplainer",
    "MarkdownRenderer",
    "MockLLMProvider",
    "OllamaExplainer",
    "OpenAIExplainer",
    "OpenRouterExplainer",
    "PDFRenderer",
    "PandasSource",
    "ParquetSource",
    "Pipeline",
    "PluginError",
    "ProphetDetector",
    "RSSFeedCollector",
    "RedditCollector",
    "RiverDetector",
    "RupturesDetector",
    "StackOverflowCollector",
    "UnknownPluginError",
    "WikipediaCollector",
    "ZScoreDetector",
    "__version__",
    "list_plugins",
    "load_plugin",
]
