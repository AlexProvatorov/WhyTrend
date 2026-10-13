"""``whytrend analyze`` command."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

from whytrend.cli.factory import (
    COLLECTOR_ALIASES,
    UsageError,
    build_pipeline,
    format_plugin_list,
    parse_collector_list,
    parse_setting,
    require_plugin,
)
from whytrend.core.models import Report
from whytrend.report import HTMLRenderer, JSONRenderer, MarkdownRenderer, PDFRenderer

_EXAMPLES = """\
examples:
  whytrend analyze --csv path.csv --keyword Python --detector zscore \\
      --explainer mock --out report.md
  whytrend analyze --trends Python --explainer llm --provider mock \\
      --format html --out report.html
  whytrend analyze --csv path.csv --collectors rss \\
      --set collector.rss.feed_urls=https://example.com/feed \\
      --format pdf --out report.pdf
"""

_TEXT_RENDERERS: dict[str, Callable[[Report], str]] = {
    "markdown": MarkdownRenderer().render,
    "json": JSONRenderer().render,
    "html": HTMLRenderer().render,
}
_FORMATS = [*_TEXT_RENDERERS, "pdf"]


class AnalyzeCommand:
    """Run detection and explanation, then write a report."""

    name = "analyze"

    def add_parser(self, subparsers: Any) -> None:
        """Register the ``analyze`` subcommand."""
        parser = subparsers.add_parser(
            self.name,
            help="Run analysis and write a report",
            description="Load a series, detect changes, and write a report.",
            epilog=_EXAMPLES,
            formatter_class=argparse.RawDescriptionHelpFormatter,
        )
        source = parser.add_mutually_exclusive_group(required=True)
        source.add_argument("--csv", type=Path, metavar="PATH", help="CSV time series file")
        source.add_argument(
            "--parquet",
            type=Path,
            metavar="PATH",
            help="Parquet time series file",
        )
        source.add_argument(
            "--trends",
            metavar="KEYWORD",
            help="Google Trends keyword",
        )
        parser.add_argument(
            "--keyword",
            help="Series keyword. Defaults to the file name",
        )
        parser.add_argument(
            "--time-col",
            default="timestamp",
            help="Timestamp column (default: timestamp)",
        )
        parser.add_argument(
            "--value-col",
            default="value",
            help="Value column (default: value)",
        )
        parser.add_argument(
            "--window-days",
            type=int,
            default=3,
            help="Days of context around each event (default: 3)",
        )
        parser.add_argument(
            "--detector",
            default="zscore",
            type=_plugin_arg("detectors"),
            metavar="NAME",
            help=f"Detector plugin (default: zscore). Available: {_names('detectors')}",
        )
        parser.add_argument(
            "--explainer",
            default="mock",
            type=_plugin_arg("explainers"),
            metavar="NAME",
            help=f"Explainer plugin (default: mock). Available: {_names('explainers')}",
        )
        parser.add_argument(
            "--provider",
            type=_plugin_arg("llm"),
            metavar="NAME",
            help=f"LLM provider for --explainer llm. Available: {_names('llm')}",
        )
        parser.add_argument(
            "--collectors",
            type=_collectors_arg,
            metavar="NAME[,NAME...]",
            help=(
                "Comma-separated collector plugins. "
                f"Alias: {_alias_help()}. Available: {_names('collectors')}"
            ),
        )
        parser.add_argument(
            "--ranker",
            type=_plugin_arg("rankers"),
            metavar="NAME",
            help=f"Optional ranker plugin. Available: {_names('rankers')}",
        )
        parser.add_argument(
            "--format",
            choices=sorted(_FORMATS),
            default="markdown",
            help="Report format (default: markdown). pdf requires --out",
        )
        parser.add_argument(
            "--out",
            type=Path,
            metavar="PATH",
            help="Report file. Prints to stdout when omitted",
        )
        parser.add_argument(
            "--set",
            dest="settings",
            action="append",
            metavar="TARGET.PARAM=VALUE",
            type=_setting_arg,
            help=(
                "Override a constructor parameter. "
                "Example: detector.threshold=1.0 or collector.hn.max_results=5"
            ),
        )
        parser.set_defaults(handler=self.run)

    def run(self, args: argparse.Namespace) -> int:
        """Execute analysis and write the report."""
        if args.format == "pdf" and args.out is None:
            msg = "--format pdf requires --out"
            raise UsageError(msg)
        report = build_pipeline(args).run()
        _write_report(report, args.format, args.out)
        return 0


def _names(group: str) -> str:
    return format_plugin_list(group)


def _alias_help() -> str:
    return ", ".join(f"{alias}={target}" for alias, target in sorted(COLLECTOR_ALIASES.items()))


def _plugin_arg(group: str) -> Callable[[str], str]:
    def parse(value: str) -> str:
        try:
            return require_plugin(group, value)
        except UsageError as exc:
            raise argparse.ArgumentTypeError(str(exc)) from exc

    return parse


def _collectors_arg(value: str) -> list[str]:
    try:
        return parse_collector_list(value)
    except UsageError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def _setting_arg(value: str) -> object:
    try:
        return parse_setting(value)
    except UsageError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def _write_report(report: Report, format_name: str, out: Path | None) -> None:
    if format_name == "pdf":
        if out is None:
            msg = "--format pdf requires --out"
            raise UsageError(msg)
        out.write_bytes(PDFRenderer().render(report))
        return
    text = _TEXT_RENDERERS[format_name](report)
    if not text.endswith("\n"):
        text += "\n"
    if out is None:
        sys.stdout.write(text)
        return
    out.write_text(text, encoding="utf-8")
