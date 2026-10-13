"""CLI tests. The mock explainer path does not use the network."""

from __future__ import annotations

import contextlib
import io
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from whytrend.cli import main
from whytrend.cli.factory import build_pipeline, parse_collector_list
from whytrend.cli.parser import build_parser
from whytrend.core.series import TimeSeries
from whytrend.llm import MockLLMProvider
from whytrend.plugins import load_plugin

FIXTURE_CSV = Path(__file__).resolve().parent / "fixtures" / "python_interest.csv"


def _invoke(argv: list[str]) -> tuple[int, str, str]:
    stdout = io.StringIO()
    stderr = io.StringIO()
    with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
        try:
            main(argv)
        except SystemExit as exc:
            return _exit_code(exc), stdout.getvalue(), stderr.getvalue()
    return 0, stdout.getvalue(), stderr.getvalue()


def _exit_code(exc: SystemExit) -> int:
    if exc.code is None:
        return 0
    if isinstance(exc.code, int):
        return exc.code
    return 1


def test_root_help_lists_analyze_and_example() -> None:
    code, stdout, stderr = _invoke(["--help"])

    assert code == 0
    assert stderr == ""
    assert "analyze" in stdout
    assert "--csv" in stdout


def test_analyze_help_lists_plugins_and_example() -> None:
    code, stdout, stderr = _invoke(["analyze", "--help"])

    assert code == 0
    assert stderr == ""
    collapsed = " ".join(stdout.split())
    assert "--csv" in collapsed
    assert "--detector" in collapsed
    assert "zscore" in collapsed
    assert "mock" in collapsed
    assert "path.csv" in collapsed
    assert "--trends" in collapsed
    assert "--provider" in collapsed
    assert "rss (needs feed_urls)" in collapsed
    assert "llm (needs --provider)" in collapsed


def test_module_help() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "whytrend", "--help"],
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert result.returncode == 0
    assert "analyze" in result.stdout


def test_console_script_help() -> None:
    executable = shutil.which("whytrend")
    assert executable is not None
    result = subprocess.run(
        [executable, "--help"],
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert result.returncode == 0
    assert "analyze" in result.stdout


def test_analyze_writes_markdown_report(tmp_path: Path) -> None:
    out = tmp_path / "report.md"
    code, stdout, stderr = _invoke(
        [
            "analyze",
            "--csv",
            str(FIXTURE_CSV),
            "--keyword",
            "Python",
            "--detector",
            "zscore",
            "--set",
            "detector.threshold=1.0",
            "--explainer",
            "mock",
            "--format",
            "markdown",
            "--out",
            str(out),
        ]
    )

    assert code == 0, stderr
    assert stdout == ""
    text = out.read_text(encoding="utf-8")
    assert text.startswith("# WhyTrend Report")
    assert "Python" in text


def test_analyze_writes_json_report(tmp_path: Path) -> None:
    out = tmp_path / "report.json"
    code, _stdout, stderr = _invoke(
        [
            "analyze",
            "--csv",
            str(FIXTURE_CSV),
            "--keyword",
            "Python",
            "--set",
            "detector.threshold=1.0",
            "--format",
            "json",
            "--out",
            str(out),
        ]
    )

    assert code == 0, stderr
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["keyword"] == "Python"


def test_analyze_prints_report_when_out_is_omitted() -> None:
    code, stdout, stderr = _invoke(
        [
            "analyze",
            "--csv",
            str(FIXTURE_CSV),
            "--set",
            "detector.threshold=1.0",
        ]
    )

    assert code == 0, stderr
    assert stdout.startswith("# WhyTrend Report")


def test_missing_source_exits_with_message() -> None:
    code, _stdout, stderr = _invoke(["analyze"])

    assert code == 2
    assert "--csv" in stderr
    assert "required" in stderr


def test_unknown_detector_exits_with_message() -> None:
    code, _stdout, stderr = _invoke(["analyze", "--csv", str(FIXTURE_CSV), "--detector", "nope"])

    assert code == 2
    assert "nope" in stderr


def test_invalid_set_exits_with_message() -> None:
    code, _stdout, stderr = _invoke(
        [
            "analyze",
            "--csv",
            str(FIXTURE_CSV),
            "--set",
            "detector.not_a_real_param=1",
        ]
    )

    assert code == 2
    assert "not_a_real_param" in stderr


def test_llm_without_provider_lists_providers() -> None:
    code, _stdout, stderr = _invoke(["analyze", "--csv", str(FIXTURE_CSV), "--explainer", "llm"])

    assert code == 2
    assert "--provider" in stderr
    assert "mock" in stderr


def test_provider_requires_llm_explainer() -> None:
    code, _stdout, stderr = _invoke(
        ["analyze", "--csv", str(FIXTURE_CSV), "--explainer", "mock", "--provider", "openai"]
    )

    assert code == 2
    assert "--provider" in stderr


def test_llm_mock_provider_writes_report(tmp_path: Path) -> None:
    out = tmp_path / "report.md"
    code, _stdout, stderr = _invoke(
        [
            "analyze",
            "--csv",
            str(FIXTURE_CSV),
            "--keyword",
            "Python",
            "--set",
            "detector.threshold=1.0",
            "--explainer",
            "llm",
            "--provider",
            "mock",
            "--out",
            str(out),
        ]
    )

    assert code == 0, stderr
    assert "Python" in out.read_text(encoding="utf-8")


def test_azure_explainer_requires_deployment() -> None:
    code, _stdout, stderr = _invoke(
        ["analyze", "--csv", str(FIXTURE_CSV), "--explainer", "azure_openai"]
    )

    assert code == 2
    assert "deployment" in stderr


def test_rss_repos_and_typed_settings_are_constructed() -> None:
    args = build_parser().parse_args(
        [
            "analyze",
            "--csv",
            str(FIXTURE_CSV),
            "--detector",
            "prophet",
            "--collectors",
            "rss,github_releases",
            "--set",
            "collector.rss.feed_urls=https://blog.python.org/feeds/posts/default",
            "--set",
            "collector.github_releases.repos=python/cpython,pallets/flask",
            "--set",
            'detector.prophet_kwargs={"seasonality_mode":"multiplicative"}',
            "--ranker",
            "bm25",
            "--set",
            "ranker.top_k=none",
            "--explainer",
            "azure_openai",
            "--set",
            "explainer.deployment=my-deploy",
        ]
    )
    pipeline = build_pipeline(args)
    collectors = {collector.name: collector for collector in pipeline._collectors}

    assert collectors["rss"].feed_urls == ["https://blog.python.org/feeds/posts/default"]
    assert collectors["github_releases"]._repos == ["python/cpython", "pallets/flask"]
    assert pipeline._detector._prophet_kwargs == {"seasonality_mode": "multiplicative"}
    assert pipeline._ranker is not None
    assert pipeline._ranker._top_k is None
    assert pipeline._explainer.provider._deployment == "my-deploy"


def test_trends_uses_patched_load(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    async def _load(self: Any) -> TimeSeries:
        assert self._geo == "US"
        frame = pd.read_csv(FIXTURE_CSV)
        return TimeSeries.from_dataframe(
            frame,
            time_col="timestamp",
            value_col="value",
            name="trends",
            keyword=self.keyword,
        )

    monkeypatch.setattr("whytrend.sources.google_trends.GoogleTrends.load", _load)
    out = tmp_path / "report.md"
    code, _stdout, stderr = _invoke(
        [
            "analyze",
            "--trends",
            "Python",
            "--set",
            "source.geo=US",
            "--set",
            "detector.threshold=1.0",
            "--explainer",
            "mock",
            "--out",
            str(out),
        ]
    )

    assert code == 0, stderr
    assert "Python" in out.read_text(encoding="utf-8")


def test_trends_keyword_must_match() -> None:
    code, _stdout, stderr = _invoke(["analyze", "--trends", "Python", "--keyword", "Java"])

    assert code == 2
    assert "--keyword must match --trends" in stderr


def test_analyze_writes_html_report(tmp_path: Path) -> None:
    out = tmp_path / "report.html"
    code, _stdout, stderr = _invoke(
        [
            "analyze",
            "--csv",
            str(FIXTURE_CSV),
            "--set",
            "detector.threshold=1.0",
            "--format",
            "html",
            "--out",
            str(out),
        ]
    )

    assert code == 0, stderr
    assert out.read_text(encoding="utf-8").startswith("<!DOCTYPE html>")


def test_pdf_requires_out_path() -> None:
    code, _stdout, stderr = _invoke(
        [
            "analyze",
            "--csv",
            str(FIXTURE_CSV),
            "--format",
            "pdf",
        ]
    )

    assert code == 2
    assert "--out" in stderr


def test_collector_alias_resolves_to_plugin_name() -> None:
    assert parse_collector_list("hn,wikipedia") == ["hacker_news", "wikipedia"]


def test_llm_provider_entry_point() -> None:
    assert load_plugin("llm", "mock") is MockLLMProvider
