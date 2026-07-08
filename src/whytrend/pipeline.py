"""Fluent analysis pipeline for WhyTrend."""

from __future__ import annotations

import asyncio
from typing import Self

from whytrend.core.models import Event, Evidence, Explanation, Report
from whytrend.core.protocols import Collector, Detector, Explainer, Source
from whytrend.core.series import TimeSeries
from whytrend.events.builder import EventBuilder


class Pipeline:
    """Orchestrates source loading, detection, collection, and explanation."""

    def __init__(self, *, window_days: int = 3) -> None:
        self._source: Source | None = None
        self._detector: Detector | None = None
        self._collectors: list[Collector] = []
        self._explainer: Explainer | None = None
        self._event_builder = EventBuilder(window_days=window_days)

    def add_source(self, source: Source) -> Self:
        if self._source is not None:
            msg = "source is already configured"
            raise ValueError(msg)
        self._source = source
        return self

    def add_detector(self, detector: Detector) -> Self:
        if self._detector is not None:
            msg = "detector is already configured"
            raise ValueError(msg)
        self._detector = detector
        return self

    def add_collector(self, collector: Collector) -> Self:
        self._collectors.append(collector)
        return self

    def add_explainer(self, explainer: Explainer) -> Self:
        if self._explainer is not None:
            msg = "explainer is already configured"
            raise ValueError(msg)
        self._explainer = explainer
        return self

    def run(self) -> Report:
        """Execute the pipeline synchronously."""
        return asyncio.run(self.arun())

    async def arun(self) -> Report:
        """Execute the pipeline asynchronously."""
        source = self._require_source()
        detector = self._require_detector()
        explainer = self._require_explainer()

        series = await source.load()
        detection_result = detector.detect(series)

        if not detection_result.detections:
            return self._empty_report(series)

        events = self._event_builder.build(detection_result, series)
        explanations = await asyncio.gather(
            *[self._explain_event(explainer, event) for event in events]
        )

        return Report(
            executive_summary=self._build_executive_summary(explanations),
            explanations=list(explanations),
            series_name=series.name,
            keyword=series.keyword or series.name,
            metadata={
                "detector": detector.name,
                "source": source.name,
                "explainer": explainer.name,
                "collectors": [collector.name for collector in self._collectors],
                "event_count": len(events),
            },
        )

    async def _explain_event(self, explainer: Explainer, event: Event) -> Explanation:
        evidences = await self._collect_evidences(event)
        return await explainer.explain(event, evidences)

    async def _collect_evidences(self, event: Event) -> list[Evidence]:
        if not self._collectors:
            return []

        results = await asyncio.gather(
            *[collector.collect(event) for collector in self._collectors],
            return_exceptions=True,
        )

        evidences: list[Evidence] = []
        for result in results:
            if not isinstance(result, list):
                continue
            evidences.extend(result)
        return evidences

    def _require_source(self) -> Source:
        if self._source is None:
            msg = "source is required; call add_source() before run()"
            raise ValueError(msg)
        return self._source

    def _require_detector(self) -> Detector:
        if self._detector is None:
            msg = "detector is required; call add_detector() before run()"
            raise ValueError(msg)
        return self._detector

    def _require_explainer(self) -> Explainer:
        if self._explainer is None:
            msg = "explainer is required; call add_explainer() before run()"
            raise ValueError(msg)
        return self._explainer

    @staticmethod
    def _empty_report(series: TimeSeries) -> Report:
        return Report(
            executive_summary="No anomalies detected.",
            series_name=series.name,
            keyword=series.keyword or series.name,
            metadata={"event_count": 0},
        )

    @staticmethod
    def _build_executive_summary(explanations: list[Explanation]) -> str:
        if not explanations:
            return "No anomalies detected."
        if len(explanations) == 1:
            return explanations[0].summary
        return " ".join(explanation.summary for explanation in explanations)
