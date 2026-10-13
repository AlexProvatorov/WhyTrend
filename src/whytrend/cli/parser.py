"""Root argument parser and command registry."""

from __future__ import annotations

import argparse
from collections.abc import Callable
from typing import cast

from whytrend.cli.analyze import AnalyzeCommand

_EXAMPLES = """\
examples:
  whytrend analyze --csv path.csv --keyword Python --detector zscore \\
      --explainer mock --out report.md
"""

COMMANDS = (AnalyzeCommand(),)


def build_parser() -> argparse.ArgumentParser:
    """Build the ``whytrend`` parser."""
    parser = argparse.ArgumentParser(
        prog="whytrend",
        description="Explainable time series analysis from the command line.",
        epilog=_EXAMPLES,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", metavar="command", required=True)
    for command in COMMANDS:
        command.add_parser(subparsers)
    return parser


def run_args(argv: list[str] | None = None) -> int:
    """Parse ``argv`` and run the selected command."""
    parser = build_parser()
    args = parser.parse_args(argv)
    handler = cast(Callable[[argparse.Namespace], int], args.handler)
    return handler(args)
