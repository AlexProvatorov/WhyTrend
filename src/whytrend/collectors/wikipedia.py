"""Collect context from Wikipedia search results."""

from __future__ import annotations

from typing import Any
from urllib.parse import quote

import httpx

from whytrend.collectors._http import get_http_client, get_json
from whytrend.collectors._utils import evidence_from_fields
from whytrend.core.models import Event, Evidence
from whytrend.core.protocols import BaseCollector


class WikipediaCollector(BaseCollector):
    """Search Wikipedia articles related to the event keyword."""

    def __init__(
        self,
        *,
        language: str = "en",
        timeout: float = 10.0,
        max_results: int = 10,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        if max_results < 1:
            msg = "max_results must be >= 1"
            raise ValueError(msg)
        if not language.strip():
            msg = "language cannot be empty"
            raise ValueError(msg)

        self._language = language.strip()
        self._timeout = timeout
        self._max_results = max_results
        self._client = client

    @property
    def name(self) -> str:
        return "wikipedia"

    @property
    def api_url(self) -> str:
        return f"https://{self._language}.wikipedia.org/w/api.php"

    async def collect(self, event: Event) -> list[Evidence]:
        params = {
            "action": "query",
            "list": "search",
            "srsearch": event.keyword,
            "format": "json",
            "utf8": 1,
            "srlimit": self._max_results,
        }

        async with get_http_client(timeout=self._timeout, client=self._client) as client:
            payload = await get_json(client, self.api_url, params=params)

        return self._parse_search_results(payload)

    def _parse_search_results(self, payload: dict[str, Any]) -> list[Evidence]:
        query = payload.get("query")
        if not isinstance(query, dict):
            return []

        search_results = query.get("search")
        if not isinstance(search_results, list):
            return []

        evidences: list[Evidence] = []
        for item in search_results:
            if not isinstance(item, dict):
                continue

            title = str(item.get("title") or "").strip()
            if not title:
                continue

            snippet = str(item.get("snippet") or "").strip()
            evidences.append(
                evidence_from_fields(
                    title=title,
                    url=self._article_url(title),
                    snippet=snippet,
                    source_name=self.name,
                    metadata={"pageid": item.get("pageid")},
                )
            )

        return evidences

    def _article_url(self, title: str) -> str:
        slug = quote(title.replace(" ", "_"))
        return f"https://{self._language}.wikipedia.org/wiki/{slug}"
