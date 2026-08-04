from __future__ import annotations

from whytrend.core import (
    BaseCollector,
    BaseDetector,
    BaseExplainer,
    BaseSource,
    Cause,
    Detection,
    DetectionResult,
    Event,
    Evidence,
    Explanation,
    TimeSeries,
)


class StubSource(BaseSource):
    def __init__(self, series: TimeSeries, *, name: str = "stub-source") -> None:
        self._series = series
        self._name = name

    @property
    def name(self) -> str:
        return self._name

    async def load(self) -> TimeSeries:
        return self._series


class StubDetector(BaseDetector):
    def __init__(
        self,
        detections: list[Detection] | None = None,
        *,
        name: str = "stub-detector",
    ) -> None:
        self._detections = detections or []
        self._name = name

    @property
    def name(self) -> str:
        return self._name

    def detect(self, series: TimeSeries) -> DetectionResult:
        return DetectionResult(
            detections=self._detections,
            detector_name=self._name,
            series_name=series.name,
        )


class StubCollector(BaseCollector):
    def __init__(
        self,
        evidences: list[Evidence] | None = None,
        *,
        name: str = "stub-collector",
    ) -> None:
        self._evidences = evidences or []
        self._name = name

    @property
    def name(self) -> str:
        return self._name

    async def collect(self, event: Event) -> list[Evidence]:
        _ = event
        return list(self._evidences)


class FailingCollector(BaseCollector):
    @property
    def name(self) -> str:
        return "failing-collector"

    async def collect(self, event: Event) -> list[Evidence]:
        _ = event
        msg = "collector unavailable"
        raise RuntimeError(msg)


class FailingExplainer(BaseExplainer):
    @property
    def name(self) -> str:
        return "failing-explainer"

    async def explain(self, event: Event, evidences: list[Evidence]) -> Explanation:
        _ = event, evidences
        msg = "explainer unavailable"
        raise RuntimeError(msg)


class StubExplainer(BaseExplainer):
    def __init__(self, *, name: str = "stub-explainer", summary_prefix: str = "Explained") -> None:
        self._name = name
        self._summary_prefix = summary_prefix

    @property
    def name(self) -> str:
        return self._name

    async def explain(self, event: Event, evidences: list[Evidence]) -> Explanation:
        top = evidences[0] if evidences else None
        causes = []
        if top is not None:
            causes.append(
                Cause(
                    source=top.source_name,
                    score=top.relevance_score,
                    url=top.url,
                    title=top.title,
                    summary=top.snippet,
                )
            )

        return Explanation(
            event_id=event.id,
            anomaly_type=event.anomaly_type,
            confidence=event.detection_score,
            summary=f"{self._summary_prefix} {event.keyword} {event.anomaly_type.value}.",
            causes=causes,
            metadata={"evidence_count": len(evidences)},
        )
