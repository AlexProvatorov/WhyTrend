"""Minimal plugin classes used by registry unit tests."""

from __future__ import annotations

from whytrend.core import (
    BaseCollector,
    BaseDetector,
    DetectionResult,
    Event,
    Evidence,
    TimeSeries,
)


class FakeSpikeDetector(BaseDetector):
    @property
    def name(self) -> str:
        return "fake_spike"

    def detect(self, series: TimeSeries) -> DetectionResult:
        return DetectionResult(
            detections=[],
            detector_name=self.name,
            series_name=series.name,
        )


class FakeNewsCollector(BaseCollector):
    @property
    def name(self) -> str:
        return "fake_news"

    async def collect(self, event: Event) -> list[Evidence]:
        _ = event
        return []
