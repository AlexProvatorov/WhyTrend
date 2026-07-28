"""Collect context from Stack Overflow via the Stack Exchange API."""

from __future__ import annotations

import os
import re
from datetime import UTC, datetime
from typing import Any

import httpx

from whytrend.collectors._http import get_http_client
from whytrend.collectors._utils import evidence_from_fields, unix_to_datetime
from whytrend.core.models import Event, Evidence
from whytrend.core.protocols import BaseCollector

STACKEXCHANGE_API_BASE = "https://api.stackexchange.com/2.3"
DEFAULT_SITE = "stackoverflow"
_HTML_TAG_PATTERN = re.compile(r"<[^>]+>")


class StackOverflowCollector(BaseCollector):
    """Search Stack Overflow questions around an event time window.

    Uses the Stack Exchange ``/search/advanced`` endpoint. Works without a
    key for light use; set ``api_key`` or ``STACKEXCHANGE_KEY`` for a higher
    daily quota (https://stackapps.com/).
    """

    def __init__(
        self,
        *,
        api_key: str | None = None,
        site: str = DEFAULT_SITE,
        timeout: float = 10.0,
        max_results: int = 10,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        if max_results < 1:
            msg = "max_results must be >= 1"
            raise ValueError(msg)
        if not site.strip():
            msg = "site cannot be empty"
            raise ValueError(msg)

        self._api_key = (api_key or os.getenv("STACKEXCHANGE_KEY") or "").strip() or None
        self._site = site.strip()
        self._timeout = timeout
        self._max_results = max_results
        self._client = client

    @property
    def name(self) -> str:
        return "stack_overflow"

    async def collect(self, event: Event) -> list[Evidence]:
        params: dict[str, Any] = {
            "q": event.keyword,
            "site": self._site,
            "sort": "relevance",
            "order": "desc",
            "pagesize": min(100, self._max_results),
            "fromdate": int(self._as_utc(event.window_start).timestamp()),
            "todate": int(self._as_utc(event.window_end).timestamp()),
            "filter": "withbody",
        }
        if self._api_key is not None:
            params["key"] = self._api_key

        async with get_http_client(timeout=self._timeout, client=self._client) as client:
            try:
                response = await client.get(
                    f"{STACKEXCHANGE_API_BASE}/search/advanced",
                    params=params,
                )
                if response.status_code in {400, 403, 429}:
                    return []
                response.raise_for_status()
                payload = response.json()
            except (httpx.HTTPError, ValueError, TypeError):
                return []

        if not isinstance(payload, dict):
            return []
        return self._parse_items(payload.get("items", []))

    def _parse_items(self, items: Any) -> list[Evidence]:
        if not isinstance(items, list):
            return []

        evidences: list[Evidence] = []
        for item in items:
            evidence = self._parse_item(item)
            if evidence is None:
                continue
            evidences.append(evidence)
            if len(evidences) >= self._max_results:
                break
        return evidences

    def _parse_item(self, item: Any) -> Evidence | None:
        if not isinstance(item, dict):
            return None

        title = str(item.get("title") or "").strip()
        link = str(item.get("link") or "").strip()
        if not title or not link:
            return None

        created = item.get("creation_date")
        published_at = None
        if created is not None:
            try:
                published_at = unix_to_datetime(int(created))
            except (TypeError, ValueError):
                published_at = None

        body = str(item.get("body") or item.get("body_markdown") or "").strip()
        snippet = self._strip_html(body)[:500] if body else title

        tags = item.get("tags")
        metadata: dict[str, Any] = {"site": self._site}
        if isinstance(tags, list):
            metadata["tags"] = [str(tag) for tag in tags]
        question_id = item.get("question_id")
        if question_id is not None:
            metadata["question_id"] = question_id
        score = item.get("score")
        if isinstance(score, (int, float)):
            metadata["score"] = int(score)

        return evidence_from_fields(
            title=title,
            url=link,
            snippet=snippet,
            source_name=self.name,
            published_at=published_at,
            metadata=metadata,
        )

    @staticmethod
    def _strip_html(value: str) -> str:
        return _HTML_TAG_PATTERN.sub("", value).strip()

    @staticmethod
    def _as_utc(moment: datetime) -> datetime:
        if moment.tzinfo is None:
            return moment.replace(tzinfo=UTC)
        return moment.astimezone(UTC)
