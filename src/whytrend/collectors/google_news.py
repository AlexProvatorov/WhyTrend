"""Collect context from Google News RSS search results."""

from __future__ import annotations

import re
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
from typing import Any
from urllib.parse import quote_plus
from xml.etree import ElementTree

import httpx

from whytrend.collectors._http import get_http_client
from whytrend.collectors._utils import evidence_from_fields
from whytrend.core.models import Event, Evidence
from whytrend.core.protocols import BaseCollector

GOOGLE_NEWS_RSS_URL = "https://news.google.com/rss/search"
_HTML_TAG_PATTERN = re.compile(r"<[^>]+>")


class GoogleNewsCollector(BaseCollector):
    """Search Google News articles related to the event keyword."""

    def __init__(
        self,
        *,
        language: str = "en",
        region: str = "US",
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
        if not region.strip():
            msg = "region cannot be empty"
            raise ValueError(msg)

        self._language = language.strip()
        self._region = region.strip().upper()
        self._timeout = timeout
        self._max_results = max_results
        self._client = client

    @property
    def name(self) -> str:
        return "google_news"

    def rss_url(self, keyword: str) -> str:
        """Build the Google News RSS search URL for ``keyword``."""
        params = {
            "q": keyword,
            "hl": self._hl,
            "gl": self._region,
            "ceid": f"{self._region}:{self._language}",
        }
        query = "&".join(f"{key}={quote_plus(str(value))}" for key, value in params.items())
        return f"{GOOGLE_NEWS_RSS_URL}?{query}"

    @property
    def _hl(self) -> str:
        if "-" in self._language:
            return self._language
        return f"{self._language}-{self._region}"

    async def collect(self, event: Event) -> list[Evidence]:
        url = self.rss_url(event.keyword)

        async with get_http_client(timeout=self._timeout, client=self._client) as client:
            response = await client.get(url)
            response.raise_for_status()

        return self._parse_rss(
            response.text,
            window_start=event.window_start,
            window_end=event.window_end,
        )

    def _parse_rss(
        self,
        payload: str,
        *,
        window_start: datetime,
        window_end: datetime,
    ) -> list[Evidence]:
        try:
            root = ElementTree.fromstring(payload)
        except ElementTree.ParseError:
            return []

        evidences: list[Evidence] = []
        for item in root.iter("item"):
            evidence = self._parse_item(
                item,
                window_start=window_start,
                window_end=window_end,
            )
            if evidence is not None:
                evidences.append(evidence)
            if len(evidences) >= self._max_results:
                break

        return evidences

    def _parse_item(
        self,
        item: ElementTree.Element,
        *,
        window_start: datetime,
        window_end: datetime,
    ) -> Evidence | None:
        title = self._element_text(item, "title")
        url = self._element_text(item, "link")
        if not title or not url:
            return None

        published_at = self._parse_pub_date(self._element_text(item, "pubDate"))
        if published_at is not None and not self._in_window(
            published_at,
            window_start=window_start,
            window_end=window_end,
        ):
            return None

        snippet = self._clean_snippet(self._element_text(item, "description"))
        source = self._element_text(item, "source")

        metadata: dict[str, Any] = {}
        if source:
            metadata["publisher"] = source

        return evidence_from_fields(
            title=title,
            url=url,
            snippet=snippet[:500],
            source_name=self.name,
            published_at=published_at,
            metadata=metadata,
        )

    @staticmethod
    def _element_text(parent: ElementTree.Element, tag: str) -> str:
        element = parent.find(tag)
        if element is None or element.text is None:
            return ""
        return element.text.strip()

    @staticmethod
    def _parse_pub_date(value: str) -> datetime | None:
        if not value:
            return None
        try:
            parsed = parsedate_to_datetime(value)
        except (TypeError, ValueError, IndexError):
            return None
        if parsed.tzinfo is None:
            return parsed.replace(tzinfo=UTC)
        return parsed.astimezone(UTC)

    @staticmethod
    def _in_window(
        published_at: datetime,
        *,
        window_start: datetime,
        window_end: datetime,
    ) -> bool:
        start = GoogleNewsCollector._as_utc(window_start)
        end = GoogleNewsCollector._as_utc(window_end)
        moment = GoogleNewsCollector._as_utc(published_at)
        return start <= moment <= end

    @staticmethod
    def _as_utc(moment: datetime) -> datetime:
        if moment.tzinfo is None:
            return moment.replace(tzinfo=UTC)
        return moment.astimezone(UTC)

    @staticmethod
    def _clean_snippet(value: str) -> str:
        if not value:
            return ""
        return _HTML_TAG_PATTERN.sub("", value).strip()
