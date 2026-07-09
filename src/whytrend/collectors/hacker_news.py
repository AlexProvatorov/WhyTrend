"""Collect context from Hacker News stories."""

from __future__ import annotations

from typing import Any

import httpx

from whytrend.collectors._http import get_http_client, get_json
from whytrend.collectors._utils import evidence_from_fields, unix_to_datetime
from whytrend.core.models import Event, Evidence
from whytrend.core.protocols import BaseCollector

HN_SEARCH_URL = "https://hn.algolia.com/api/v1/search"


class HackerNewsCollector(BaseCollector):
    """Search Hacker News stories around an event time window."""

    def __init__(
        self,
        *,
        timeout: float = 10.0,
        max_results: int = 10,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        if max_results < 1:
            msg = "max_results must be >= 1"
            raise ValueError(msg)

        self._timeout = timeout
        self._max_results = max_results
        self._client = client

    @property
    def name(self) -> str:
        return "hacker_news"

    async def collect(self, event: Event) -> list[Evidence]:
        params = {
            "query": event.keyword,
            "tags": "story",
            "numericFilters": (
                f"created_at_i>={int(event.window_start.timestamp())},"
                f"created_at_i<={int(event.window_end.timestamp())}"
            ),
            "hitsPerPage": self._max_results,
        }

        async with get_http_client(timeout=self._timeout, client=self._client) as client:
            payload = await get_json(client, HN_SEARCH_URL, params=params)

        return self._parse_hits(payload.get("hits", []))

    def _parse_hits(self, hits: Any) -> list[Evidence]:
        if not isinstance(hits, list):
            return []

        evidences: list[Evidence] = []
        for hit in hits:
            if not isinstance(hit, dict):
                continue

            title = str(hit.get("title") or hit.get("story_title") or "").strip()
            if not title:
                continue

            object_id = hit.get("objectID") or hit.get("story_id")
            url = str(hit.get("url") or hit.get("story_url") or "").strip()
            if not url and object_id is not None:
                url = f"https://news.ycombinator.com/item?id={object_id}"
            if not url:
                continue

            snippet = str(hit.get("story_text") or hit.get("comment_text") or "").strip()
            created_at = hit.get("created_at_i")
            published_at = unix_to_datetime(int(created_at)) if created_at is not None else None

            evidences.append(
                evidence_from_fields(
                    title=title,
                    url=url,
                    snippet=snippet[:500],
                    source_name=self.name,
                    published_at=published_at,
                    metadata={"object_id": str(object_id) if object_id is not None else None},
                )
            )

        return evidences
