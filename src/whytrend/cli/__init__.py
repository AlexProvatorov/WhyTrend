"""Command-line interface for WhyTrend."""

from __future__ import annotations

import sys

from whytrend.cli.factory import UsageError
from whytrend.cli.parser import run_args


def main(argv: list[str] | None = None) -> None:
    """Parse arguments, run the selected command, and exit with its status code."""
    raise SystemExit(_status(argv))


def _status(argv: list[str] | None) -> int:
    try:
        return run_args(argv)
    except UsageError as exc:
        print(exc, file=sys.stderr)
        return exc.code
    except Exception as exc:
        print(exc, file=sys.stderr)
        return 1
