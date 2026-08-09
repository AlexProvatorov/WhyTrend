"""Bitcoin 2022–2026 case study: regime shifts → context → report.

Uses daily BTCUSDT closes from Binance (see ``examples/data/``).

Why these tools
---------------
- ``RupturesDetector`` — structural breaks / regime changes (not every noisy spike)
- ``HackerNewsCollector`` — historical search with a real time window (unlike live News RSS)
- ``BM25Ranker`` — rank HN hits for the event keyword
- ``MockLLMProvider`` by default — reproducible offline explanations for the article draft
- ``--llm ollama`` — local model (default ``qwen2.5-coder``; override with ``--ollama-model``)
- ``--llm openai`` — cloud (needs ``OPENAI_API_KEY``)

Run::

    uv sync --extra ruptures
    uv run python examples/btc_case_study_2022_2026.py
    uv run python examples/btc_case_study_2022_2026.py --llm ollama --ollama-model qwen2.5-coder:14b
    # or heavier: --ollama-model qwen2.5-coder:32b
    uv run python examples/btc_case_study_2022_2026.py --llm openai
"""

from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

import pandas as pd

from whytrend import (
    BM25Ranker,
    CSVSource,
    LLMExplainer,
    MockLLMProvider,
    OllamaExplainer,
    OpenAIExplainer,
    Pipeline,
    RupturesDetector,
    ZScoreDetector,
)
from whytrend.collectors import HackerNewsCollector
from whytrend.report import JSONRenderer, MarkdownRenderer
from whytrend.sources.pandas_source import PandasSource

ROOT = Path(__file__).resolve().parent
DATA_CSV = ROOT / "data" / "btc_usd_daily_2022_2026.csv"
OUT_DIR = ROOT / "output"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--csv",
        type=Path,
        default=DATA_CSV,
        help="Daily BTC close CSV (timestamp,value[,volume])",
    )
    parser.add_argument(
        "--penalty",
        type=float,
        default=10.0,
        help="Ruptures PELT penalty (higher → fewer changepoints)",
    )
    parser.add_argument(
        "--window-days",
        type=int,
        default=5,
        help="Context window around each changepoint (± days)",
    )
    parser.add_argument(
        "--llm",
        choices=("mock", "ollama", "openai"),
        default="mock",
        help="Explainer backend",
    )
    parser.add_argument(
        "--ollama-model",
        default="qwen2.5-coder:14b",
        help="Ollama model name when --llm ollama",
    )
    parser.add_argument(
        "--ollama-base-url",
        default="http://localhost:11434",
        help="Ollama base URL",
    )
    parser.add_argument(
        "--ollama-timeout",
        type=float,
        default=180.0,
        help="Ollama request timeout in seconds",
    )
    parser.add_argument(
        "--skip-collectors",
        action="store_true",
        help="Detect + explain without network collectors",
    )
    parser.add_argument(
        "--returns-threshold",
        type=float,
        default=3.5,
        help="Z-score threshold on daily %% returns (side report)",
    )
    return parser.parse_args()


async def _side_report_returns(csv_path: Path, threshold: float) -> None:
    """Extra pass: spike/drop days on returns (not used by the main pipeline)."""
    frame = pd.read_csv(csv_path, parse_dates=["timestamp"])
    frame = frame.sort_values("timestamp")
    returns = frame[["timestamp"]].copy()
    returns["value"] = frame["value"].pct_change() * 100.0
    returns = returns.dropna().reset_index(drop=True)

    series = await PandasSource(
        returns,
        name="btc_daily_return_pct",
        keyword="Bitcoin",
    ).load()
    result = ZScoreDetector(threshold=threshold).detect(series)
    print(f"\n## Side pass: ZScore on daily returns (threshold={threshold})")
    print(f"detections={len(result.detections)}")
    for detection in sorted(result.detections, key=lambda item: -item.score)[:12]:
        print(
            f"  {detection.timestamp.date()}  "
            f"{detection.anomaly_type.value:5}  "
            f"ret={detection.value:+.2f}%  score={detection.score:.3f}"
        )


async def main() -> None:
    args = _parse_args()
    if not args.csv.is_file():
        msg = f"CSV not found: {args.csv}. Re-fetch with examples/fetch_btc_daily.py"
        raise FileNotFoundError(msg)

    await _side_report_returns(args.csv, args.returns_threshold)

    if args.llm == "openai":
        explainer = OpenAIExplainer()
    elif args.llm == "ollama":
        explainer = OllamaExplainer(
            model=args.ollama_model,
            base_url=args.ollama_base_url,
            timeout=args.ollama_timeout,
        )
    else:
        explainer = LLMExplainer(MockLLMProvider(summary_prefix="BTC case"))

    source = CSVSource(
        args.csv,
        name="btc_usd_daily",
        keyword="Bitcoin",
        metadata={"asset": "BTCUSDT", "venue": "binance", "interval": "1d"},
    )
    detector = RupturesDetector(algorithm="pelt", model="rbf", penalty=args.penalty)

    print(f"\n## Main pipeline: Ruptures(penalty={args.penalty}) → HN → BM25 → {explainer.name}")
    series = await source.load()
    detection_result = detector.detect(series)
    print(f"changepoints={len(detection_result.detections)}")
    for detection in detection_result.detections:
        print(
            f"  {detection.timestamp.date()}  "
            f"close={detection.value:,.0f}  "
            f"score={detection.score:.3f}  "
            f"{detection.anomaly_type.value}"
        )

    pipeline = (
        Pipeline(window_days=args.window_days)
        .add_source(source)
        .add_detector(detector)
        .add_ranker(BM25Ranker(top_k=8))
        .add_explainer(explainer)
    )
    if not args.skip_collectors:
        pipeline.add_collector(HackerNewsCollector(max_results=15))

    report = await pipeline.arun()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    md_path = OUT_DIR / "btc_case_study_report.md"
    json_path = OUT_DIR / "btc_case_study_report.json"
    md_path.write_text(MarkdownRenderer().render(report), encoding="utf-8")
    json_path.write_text(JSONRenderer().render(report), encoding="utf-8")

    print(f"events={report.metadata.get('event_count')}")
    print(f"wrote {md_path}")
    print(f"wrote {json_path}")
    print()
    print(report.executive_summary)
    print()
    for explanation in report.explanations:
        print("---")
        print(explanation.summary)
        print(f"confidence={explanation.confidence:.2f} causes={len(explanation.causes)}")
        if explanation.causes:
            top = explanation.causes[0]
            print(f"top_cause={top.title!r} ({top.source})")


if __name__ == "__main__":
    asyncio.run(main())
