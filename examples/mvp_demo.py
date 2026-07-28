"""Minimal end-to-end WhyTrend demo using local fixtures and mock LLM."""

from __future__ import annotations

import asyncio
from pathlib import Path

from whytrend import (
    BM25Ranker,
    CSVSource,
    LLMExplainer,
    MockLLMProvider,
    Pipeline,
    ZScoreDetector,
)
from whytrend.core import Event, Evidence
from whytrend.report import JSONRenderer, MarkdownRenderer

FIXTURE_CSV = Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "python_interest.csv"


class DemoCollector:
    """Tiny inline collector for the MVP demo."""

    @property
    def name(self) -> str:
        return "demo"

    async def collect(self, event: Event) -> list[Evidence]:
        return [
            Evidence(
                title=f"{event.keyword} 3.13 released",
                url="https://example.com/python-3-13",
                snippet="Demo evidence for the MVP pipeline.",
                source_name=self.name,
                relevance_score=0.9,
            )
        ]


async def main() -> None:
    pipeline = (
        Pipeline(window_days=3)
        .add_source(CSVSource(FIXTURE_CSV, name="python_interest", keyword="Python"))
        .add_detector(ZScoreDetector(threshold=1.0))
        .add_collector(DemoCollector())
        .add_ranker(BM25Ranker(top_k=5))
        .add_explainer(LLMExplainer(MockLLMProvider()))
    )

    report = await pipeline.arun()

    print(MarkdownRenderer().render(report))
    print("--- JSON ---")
    print(JSONRenderer().render(report))


if __name__ == "__main__":
    asyncio.run(main())
