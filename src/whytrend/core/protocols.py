"""Component interfaces for the WhyTrend plugin architecture."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Protocol, runtime_checkable

from whytrend.core.models import DetectionResult, Event, Evidence, Explanation, Report
from whytrend.core.series import TimeSeries


@runtime_checkable
class Source(Protocol):
    """Loads or fetches a time series."""

    @property
    def name(self) -> str:
        """Human-readable source name."""
        ...

    async def load(self) -> TimeSeries:
        """Fetch and return a :class:`TimeSeries`."""
        ...


@runtime_checkable
class Detector(Protocol):
    """Detects anomalies and structural changes."""

    @property
    def name(self) -> str:
        """Human-readable detector name."""
        ...

    def detect(self, series: TimeSeries) -> DetectionResult:
        """Run detection on ``series``."""
        ...


@runtime_checkable
class Collector(Protocol):
    """Gathers external context for an event."""

    @property
    def name(self) -> str:
        """Human-readable collector name."""
        ...

    async def collect(self, event: Event) -> list[Evidence]:
        """Return evidence items related to ``event``."""
        ...


@runtime_checkable
class Ranker(Protocol):
    """Scores and orders evidence by relevance to an event."""

    @property
    def name(self) -> str:
        """Human-readable ranker name."""
        ...

    def rank(self, event: Event, evidences: list[Evidence]) -> list[Evidence]:
        """Return evidences sorted by descending relevance."""
        ...


@runtime_checkable
class LLMProvider(Protocol):
    """Generates natural-language explanations from ranked evidence."""

    @property
    def name(self) -> str:
        """Human-readable provider name."""
        ...

    async def explain(self, event: Event, evidences: list[Evidence]) -> Explanation:
        """Produce a structured explanation for ``event``."""
        ...


@runtime_checkable
class Explainer(Protocol):
    """High-level component that orchestrates ranking and explanation."""

    @property
    def name(self) -> str:
        """Human-readable explainer name."""
        ...

    async def explain(self, event: Event, evidences: list[Evidence]) -> Explanation:
        """Produce a structured explanation for ``event``."""
        ...


@runtime_checkable
class ReportRenderer(Protocol):
    """Renders a :class:`Report` to a target format."""

    @property
    def name(self) -> str:
        """Human-readable renderer name."""
        ...

    def render(self, report: Report) -> str:
        """Render ``report`` as a string (JSON, Markdown, HTML, ...)."""
        ...


class BaseSource(ABC):
    """Optional ABC for source implementations."""

    @property
    @abstractmethod
    def name(self) -> str:
        raise NotImplementedError

    @abstractmethod
    async def load(self) -> TimeSeries:
        raise NotImplementedError


class BaseDetector(ABC):
    """Optional ABC for detector implementations."""

    @property
    @abstractmethod
    def name(self) -> str:
        raise NotImplementedError

    @abstractmethod
    def detect(self, series: TimeSeries) -> DetectionResult:
        raise NotImplementedError


class BaseCollector(ABC):
    """Optional ABC for collector implementations."""

    @property
    @abstractmethod
    def name(self) -> str:
        raise NotImplementedError

    @abstractmethod
    async def collect(self, event: Event) -> list[Evidence]:
        raise NotImplementedError


class BaseRanker(ABC):
    """Optional ABC for ranker implementations."""

    @property
    @abstractmethod
    def name(self) -> str:
        raise NotImplementedError

    @abstractmethod
    def rank(self, event: Event, evidences: list[Evidence]) -> list[Evidence]:
        raise NotImplementedError


class BaseLLMProvider(ABC):
    """Optional ABC for LLM provider implementations."""

    @property
    @abstractmethod
    def name(self) -> str:
        raise NotImplementedError

    @abstractmethod
    async def explain(self, event: Event, evidences: list[Evidence]) -> Explanation:
        raise NotImplementedError


class BaseExplainer(ABC):
    """Optional ABC for explainer implementations."""

    @property
    @abstractmethod
    def name(self) -> str:
        raise NotImplementedError

    @abstractmethod
    async def explain(self, event: Event, evidences: list[Evidence]) -> Explanation:
        raise NotImplementedError


class BaseReportRenderer(ABC):
    """Optional ABC for report renderer implementations."""

    @property
    @abstractmethod
    def name(self) -> str:
        raise NotImplementedError

    @abstractmethod
    def render(self, report: Report) -> str:
        raise NotImplementedError
