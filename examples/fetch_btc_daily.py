"""Fetch daily BTCUSDT closes from Binance into examples/data/.

Default range: 2022-01-01 → 2026-08-07 (inclusive).

    uv run python examples/fetch_btc_daily.py
"""

from __future__ import annotations

import argparse
import csv
from datetime import UTC, datetime
from pathlib import Path

import httpx

DEFAULT_OUT = Path(__file__).resolve().parent / "data" / "btc_usd_daily_2022_2026.csv"
URL = "https://api.binance.com/api/v3/klines"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", default="2022-01-01")
    parser.add_argument("--end", default="2026-08-07")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    start = datetime.fromisoformat(args.start).replace(tzinfo=UTC)
    end = datetime.fromisoformat(args.end).replace(tzinfo=UTC)
    start_ms = int(start.timestamp() * 1000)
    end_ms = int(end.timestamp() * 1000)

    rows: list[tuple[str, float, float]] = []
    cursor = start_ms
    with httpx.Client(timeout=30.0) as client:
        while cursor < end_ms:
            response = client.get(
                URL,
                params={
                    "symbol": "BTCUSDT",
                    "interval": "1d",
                    "startTime": cursor,
                    "endTime": end_ms,
                    "limit": 1000,
                },
            )
            response.raise_for_status()
            batch = response.json()
            if not batch:
                break
            for candle in batch:
                open_time = datetime.fromtimestamp(candle[0] / 1000, tz=UTC)
                if start <= open_time <= end:
                    rows.append(
                        (
                            open_time.date().isoformat(),
                            float(candle[4]),
                            float(candle[5]),
                        )
                    )
            next_cursor = int(batch[-1][0]) + 86_400_000
            if next_cursor <= cursor:
                break
            cursor = next_cursor

    by_date = {date: (date, close, volume) for date, close, volume in rows}
    ordered = [by_date[date] for date in sorted(by_date)]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["timestamp", "value", "volume"])
        for date, close, volume in ordered:
            writer.writerow([date, f"{close:.2f}", f"{volume:.8f}"])

    print(f"wrote {args.out} rows={len(ordered)} {ordered[0][0]} → {ordered[-1][0]}")


if __name__ == "__main__":
    main()
