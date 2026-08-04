"""Shared parsing helpers for collectors."""

from __future__ import annotations

import re
from datetime import UTC, datetime
from typing import Any
from urllib.parse import quote

from whytrend.core.models import Evidence


def unix_to_datetime(timestamp: int) -> datetime:
    return datetime.fromtimestamp(timestamp, tz=UTC)


def wikipedia_article_url(title: str) -> str:
    return f"https://en.wikipedia.org/wiki/{quote(title.replace(' ', '_'))}"


def keyword_matches(haystack: str, keyword: str) -> bool:
    cleaned = keyword.strip()
    if not cleaned:
        return True
    pattern = rf"(?<!\w){re.escape(cleaned.casefold())}(?!\w)"
    return re.search(pattern, haystack.casefold()) is not None


def evidence_from_fields(
    *,
    title: str,
    url: str,
    snippet: str,
    source_name: str,
    published_at: datetime | None = None,
    metadata: dict[str, Any] | None = None,
) -> Evidence:
    return Evidence(
        title=title,
        url=url,
        snippet=snippet,
        source_name=source_name,
        published_at=published_at,
        metadata=dict(metadata or {}),
    )
