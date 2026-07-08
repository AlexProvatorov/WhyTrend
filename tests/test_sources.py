from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

from whytrend.sources import CSVSource, GoogleTrends, PandasSource, ParquetSource

FIXTURES_DIR = Path(__file__).parent / "fixtures"
SPIKE_CSV = FIXTURES_DIR / "python_interest.csv"


@pytest.fixture
def spike_csv_path() -> Path:
    return SPIKE_CSV


@pytest.mark.asyncio
async def test_csv_source_loads_series(spike_csv_path: Path) -> None:
    source = CSVSource(
        spike_csv_path,
        name="python_interest",
        keyword="Python",
    )

    series = await source.load()

    assert source.name == "csv"
    assert series.name == "python_interest"
    assert series.keyword == "Python"
    assert len(series) == 5
    assert series.metadata["source"] == "csv"
    assert series.data.iloc[-1] == 640.0


@pytest.mark.asyncio
async def test_csv_source_raises_when_file_missing(tmp_path: Path) -> None:
    source = CSVSource(tmp_path / "missing.csv")

    with pytest.raises(FileNotFoundError, match="CSV file not found"):
        await source.load()


@pytest.mark.asyncio
async def test_pandas_source_loads_series(spike_dataframe: pd.DataFrame) -> None:
    source = PandasSource(
        spike_dataframe,
        name="python_interest",
        keyword="Python",
    )

    series = await source.load()

    assert source.name == "pandas"
    assert series.name == "python_interest"
    assert series.keyword == "Python"
    assert len(series) == 5


@pytest.mark.asyncio
async def test_parquet_source_loads_series(spike_dataframe: pd.DataFrame, tmp_path: Path) -> None:
    path = tmp_path / "python_interest.parquet"
    spike_dataframe.to_parquet(path, index=False)

    source = ParquetSource(
        path,
        name="python_interest",
        keyword="Python",
    )

    series = await source.load()

    assert source.name == "parquet"
    assert series.name == "python_interest"
    assert series.keyword == "Python"
    assert len(series) == 5
    assert series.metadata["path"] == str(path)


@pytest.mark.asyncio
async def test_google_trends_loads_series() -> None:
    timestamps = pd.date_range("2026-01-01", periods=3, freq="D", tz="UTC")
    fake_frame = pd.DataFrame(
        {"Python": [40, 42, 95], "isPartial": [False, False, False]},
        index=timestamps,
    )

    class FakeTrendReq:
        def build_payload(self, keywords, timeframe: str, geo: str) -> None:
            assert keywords == ["Python"]
            assert timeframe == "today 12-m"
            assert geo == ""

        def interest_over_time(self) -> pd.DataFrame:
            return fake_frame

    with patch("pytrends.request.TrendReq", FakeTrendReq):
        source = GoogleTrends("Python")
        series = await source.load()

    assert source.name == "google_trends"
    assert series.keyword == "Python"
    assert len(series) == 3
    assert series.data.iloc[-1] == 95.0
    assert series.metadata["timeframe"] == "today 12-m"


def test_google_trends_rejects_empty_keyword() -> None:
    with pytest.raises(ValueError, match="keyword cannot be empty"):
        GoogleTrends("  ")
