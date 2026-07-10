"""WhyTrend — Explainable Time Series Analysis Framework."""

from whytrend.collectors import HackerNewsCollector, WikipediaCollector
from whytrend.detectors import ProphetDetector, ZScoreDetector
from whytrend.explainers import LLMExplainer, MockLLMProvider, OllamaExplainer, OpenAIExplainer
from whytrend.rankers import BM25Ranker, EmbeddingRanker
from whytrend.pipeline import Pipeline
from whytrend.report import JSONRenderer, MarkdownRenderer
from whytrend.sources import CSVSource, GoogleTrends, PandasSource, ParquetSource

__version__ = "0.1.0"
__author__ = "Alexander Provatorov"

__all__ = [
    "BM25Ranker",
    "CSVSource",
    "EmbeddingRanker",
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
    "WikipediaCollector",
    "ZScoreDetector",
    "__version__",
]
