"""Fluent analysis pipeline for WhyTrend."""

from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Self

from whytrend.core.enums import EvidenceVerdict
from whytrend.core.models import DetectionResult, Event, Evidence, Explanation, Report
from whytrend.core.protocols import Collector, Detector, Explainer, Filter, Ranker, Source
from whytrend.core.series import TimeSeries
from whytrend.events.builder import EventBuilder
from whytrend.llm._parsing import apply_verdict, fallback_explanation, ground_causes


class Pipeline:
    """Orchestrates source loading, detection, collection, and explanation."""

    def __init__(self, *, window_days: int = 3) -> None:
        self._source: Source | None = None
        self._detector: Detector | None = None
        self._collectors: list[Collector] = []
        self._ranker: Ranker | None = None
        self._filter: Filter | None = None
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

    def add_ranker(self, ranker: Ranker) -> Self:
        if self._ranker is not None:
            msg = "ranker is already configured"
            raise ValueError(msg)
        self._ranker = ranker
        return self

    def add_filter(self, evidence_filter: Filter) -> Self:
        if self._filter is not None:
            msg = "filter is already configured"
            raise ValueError(msg)
        self._filter = evidence_filter
        return self

    def add_explainer(self, explainer: Explainer) -> Self:
        if self._explainer is not None:
            msg = "explainer is already configured"
            raise ValueError(msg)
        self._explainer = explainer
        return self

    def run(self) -> Report:
        """Execute the pipeline synchronously.

        A notebook or another running event loop cannot call ``asyncio.run``.
        In that case the pipeline runs on a worker thread.
        """
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(self.arun())
        with ThreadPoolExecutor(max_workers=1) as executor:
            return executor.submit(asyncio.run, self.arun()).result()

    async def arun(self) -> Report:
        """Execute the pipeline asynchronously."""
        source = self._require_source()
        detector = self._require_detector()
        explainer = self._require_explainer()

        series = await source.load()
        detection_result = detector.detect(series)

        if not detection_result.detections:
            return self._empty_report(series, detection_result)

        events = self._event_builder.build(detection_result, series)
        explanations = list(
            await asyncio.gather(*[self._explain_event(explainer, event) for event in events])
        )
        collector_errors = [
            error
            for explanation in explanations
            for error in explanation.metadata.get("collector_errors", [])
        ]
        metadata: dict[str, Any] = {
            "detector": detector.name,
            "source": source.name,
            "explainer": explainer.name,
            "ranker": self._ranker.name if self._ranker is not None else None,
            "filter": self._filter.name if self._filter is not None else None,
            "collectors": [collector.name for collector in self._collectors],
            "event_count": len(events),
        }
        merged = [
            int(event.metadata["merged_detections"])
            for event in events
            if isinstance(event.metadata.get("merged_detections"), int)
            and event.metadata["merged_detections"] > 1
        ]
        if merged:
            metadata["merged_detections"] = merged
        if collector_errors:
            metadata["collector_errors"] = collector_errors

        return Report(
            executive_summary=self._build_executive_summary(explanations),
            explanations=explanations,
            series_name=series.name,
            keyword=series.keyword or series.name,
            metadata=metadata,
        )

    async def _explain_event(self, explainer: Explainer, event: Event) -> Explanation:
        evidences, collector_errors = await self._collect_evidences(event)
        if self._ranker is not None:
            evidences = self._ranker.rank(event, evidences)
        verdict: EvidenceVerdict | None = None
        if self._filter is not None:
            evidences, verdict = self._filter.filter(evidences)
        try:
            explanation = await explainer.explain(event, evidences)
        except Exception as exc:
            explanation = fallback_explanation(event, evidences)
            error_metadata: dict[str, Any] = {
                **explanation.metadata,
                "explainer_error": type(exc).__name__,
                "explainer_error_message": str(exc),
            }
            raw_response = getattr(exc, "raw_response", None)
            if isinstance(raw_response, str) and raw_response.strip():
                error_metadata["explainer_raw_response"] = raw_response.strip()[:4000]
            explanation = explanation.model_copy(update={"metadata": error_metadata})
        explanation = _stamp_event(explanation, event)
        explanation = ground_causes(explanation, evidences)
        if collector_errors:
            explanation = explanation.model_copy(
                update={
                    "metadata": {
                        **explanation.metadata,
                        "collector_errors": collector_errors,
                    }
                }
            )
        if self._filter is None or verdict is None:
            return explanation
        return self._filter.cap_confidence(apply_verdict(explanation, verdict, event))

    async def _collect_evidences(
        self,
        event: Event,
    ) -> tuple[list[Evidence], list[dict[str, str]]]:
        if not self._collectors:
            return [], []

        results = await asyncio.gather(
            *[collector.collect(event) for collector in self._collectors],
            return_exceptions=True,
        )

        evidences: list[Evidence] = []
        errors: list[dict[str, str]] = []
        for collector, result in zip(self._collectors, results, strict=True):
            if isinstance(result, list):
                evidences.extend(result)
                continue
            error = result if isinstance(result, Exception) else TypeError("collector failed")
            errors.append(
                {
                    "collector": collector.name,
                    "error": type(error).__name__,
                    "message": str(error),
                }
            )
        return evidences, errors

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
    def _empty_report(series: TimeSeries, result: DetectionResult) -> Report:
        reason = result.metadata.get("reason")
        summary = "No anomalies detected."
        if reason == "insufficient_points":
            summary = "No anomalies detected: insufficient points to score the series."
        elif reason == "zero_variance":
            summary = "No anomalies detected: the series variance is zero."
        metadata: dict[str, Any] = {
            "event_count": 0,
            "detector": result.detector_name,
        }
        if isinstance(reason, str) and reason:
            metadata["reason"] = reason
        return Report(
            executive_summary=summary,
            series_name=series.name,
            keyword=series.keyword or series.name,
            metadata=metadata,
        )

    @staticmethod
    def _build_executive_summary(explanations: list[Explanation]) -> str:
        if not explanations:
            return "No anomalies detected."
        if len(explanations) == 1:
            return explanations[0].summary
        return "\n\n".join(explanation.summary for explanation in explanations)


def _stamp_event(explanation: Explanation, event: Event) -> Explanation:
    update: dict[str, Any] = {
        "timestamp": event.timestamp,
        "value": event.value,
        "window_start": event.window_start,
        "window_end": event.window_end,
    }
    merged = event.metadata.get("merged_detections")
    if isinstance(merged, int) and merged > 1:
        update["metadata"] = {**explanation.metadata, "merged_detections": merged}
    return explanation.model_copy(update=update)
