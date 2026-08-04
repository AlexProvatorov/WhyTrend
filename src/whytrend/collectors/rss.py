"""Collect context from user-provided RSS/Atom feeds."""

from __future__ import annotations

import re
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
from xml.etree import ElementTree

import httpx

from whytrend.collectors._http import get_http_client
from whytrend.collectors._utils import evidence_from_fields, keyword_matches
from whytrend.core.models import Event, Evidence
from whytrend.core.protocols import BaseCollector

_HTML_TAG_PATTERN = re.compile(r"<[^>]+>")
_ATOM_NS = {"atom": "http://www.w3.org/2005/Atom"}


class RSSFeedCollector(BaseCollector):
    """Fetch and filter entries from one or more RSS/Atom feeds.

    Filters by ``event.keyword`` (title/summary) and the event time window.
    Items without a parseable publication date are dropped.
    Uses the standard library XML parser — no extra dependencies.
    """

    def __init__(
        self,
        feed_urls: list[str] | str,
        *,
        timeout: float = 10.0,
        max_results: int = 10,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        if max_results < 1:
            msg = "max_results must be >= 1"
            raise ValueError(msg)

        urls = [feed_urls] if isinstance(feed_urls, str) else list(feed_urls)

        cleaned = [url.strip() for url in urls if url.strip()]
        if not cleaned:
            msg = "feed_urls cannot be empty"
            raise ValueError(msg)

        self._feed_urls = cleaned
        self._timeout = timeout
        self._max_results = max_results
        self._client = client

    @property
    def name(self) -> str:
        return "rss"

    @property
    def feed_urls(self) -> list[str]:
        return list(self._feed_urls)

    async def collect(self, event: Event) -> list[Evidence]:
        evidences: list[Evidence] = []

        async with get_http_client(timeout=self._timeout, client=self._client) as client:
            for feed_url in self._feed_urls:
                try:
                    response = await client.get(feed_url)
                    if response.is_error:
                        continue
                    parsed = self._parse_feed(
                        response.text,
                        feed_url=feed_url,
                        keyword=event.keyword,
                        window_start=event.window_start,
                        window_end=event.window_end,
                    )
                except (httpx.HTTPError, ElementTree.ParseError):
                    continue

                for evidence in parsed:
                    evidences.append(evidence)
                    if len(evidences) >= self._max_results:
                        return evidences

        return evidences

    def _parse_feed(
        self,
        payload: str,
        *,
        feed_url: str,
        keyword: str,
        window_start: datetime,
        window_end: datetime,
    ) -> list[Evidence]:
        try:
            root = ElementTree.fromstring(payload)
        except ElementTree.ParseError:
            return []

        tag = self._local_name(root.tag)
        if tag == "feed":
            entries = root.findall("atom:entry", _ATOM_NS) or root.findall("entry")
            return self._parse_entries(
                entries,
                feed_url=feed_url,
                keyword=keyword,
                window_start=window_start,
                window_end=window_end,
                atom=True,
            )

        items = list(root.iter("item"))
        return self._parse_entries(
            items,
            feed_url=feed_url,
            keyword=keyword,
            window_start=window_start,
            window_end=window_end,
            atom=False,
        )

    def _parse_entries(
        self,
        entries: list[ElementTree.Element],
        *,
        feed_url: str,
        keyword: str,
        window_start: datetime,
        window_end: datetime,
        atom: bool,
    ) -> list[Evidence]:
        keyword_lower = keyword.casefold()
        evidences: list[Evidence] = []

        for entry in entries:
            evidence = (
                self._parse_atom_entry(
                    entry,
                    feed_url=feed_url,
                    keyword_lower=keyword_lower,
                    window_start=window_start,
                    window_end=window_end,
                )
                if atom
                else self._parse_rss_item(
                    entry,
                    feed_url=feed_url,
                    keyword_lower=keyword_lower,
                    window_start=window_start,
                    window_end=window_end,
                )
            )
            if evidence is not None:
                evidences.append(evidence)
        return evidences

    def _parse_rss_item(
        self,
        item: ElementTree.Element,
        *,
        feed_url: str,
        keyword_lower: str,
        window_start: datetime,
        window_end: datetime,
    ) -> Evidence | None:
        title = self._element_text(item, "title")
        url = self._element_text(item, "link")
        snippet = self._clean_snippet(self._element_text(item, "description"))
        published_at = self._parse_pub_date(self._element_text(item, "pubDate"))

        return self._to_evidence(
            title=title,
            url=url,
            snippet=snippet,
            published_at=published_at,
            feed_url=feed_url,
            keyword_lower=keyword_lower,
            window_start=window_start,
            window_end=window_end,
        )

    def _parse_atom_entry(
        self,
        entry: ElementTree.Element,
        *,
        feed_url: str,
        keyword_lower: str,
        window_start: datetime,
        window_end: datetime,
    ) -> Evidence | None:
        title = self._atom_text(entry, "title")
        url = self._atom_link(entry)
        snippet = self._clean_snippet(
            self._atom_text(entry, "summary") or self._atom_text(entry, "content")
        )
        published_raw = self._atom_text(entry, "published") or self._atom_text(entry, "updated")
        published_at = self._parse_iso_datetime(published_raw) or self._parse_pub_date(
            published_raw
        )

        return self._to_evidence(
            title=title,
            url=url,
            snippet=snippet,
            published_at=published_at,
            feed_url=feed_url,
            keyword_lower=keyword_lower,
            window_start=window_start,
            window_end=window_end,
        )

    def _to_evidence(
        self,
        *,
        title: str,
        url: str,
        snippet: str,
        published_at: datetime | None,
        feed_url: str,
        keyword_lower: str,
        window_start: datetime,
        window_end: datetime,
    ) -> Evidence | None:
        if not title or not url:
            return None

        haystack = f"{title} {snippet}"
        if keyword_lower and not keyword_matches(haystack, keyword_lower):
            return None

        if published_at is None or not self._in_window(
            published_at,
            window_start=window_start,
            window_end=window_end,
        ):
            return None

        return evidence_from_fields(
            title=title,
            url=url,
            snippet=snippet[:500],
            source_name=self.name,
            published_at=published_at,
            metadata={"feed_url": feed_url},
        )

    @staticmethod
    def _element_text(parent: ElementTree.Element, tag: str) -> str:
        element = parent.find(tag)
        if element is None:
            return ""
        if element.text:
            return element.text.strip()
        # CDATA / nested text
        return "".join(element.itertext()).strip()

    @staticmethod
    def _atom_text(parent: ElementTree.Element, tag: str) -> str:
        element = parent.find(f"atom:{tag}", _ATOM_NS)
        if element is None:
            element = parent.find(tag)
        if element is None:
            return ""
        if element.text:
            return element.text.strip()
        return "".join(element.itertext()).strip()

    @staticmethod
    def _atom_link(entry: ElementTree.Element) -> str:
        for link in entry.findall("atom:link", _ATOM_NS) + entry.findall("link"):
            rel = link.attrib.get("rel", "alternate")
            href = link.attrib.get("href", "").strip()
            if href and rel in {"alternate", ""}:
                return href
        return ""

    @staticmethod
    def _local_name(tag: str) -> str:
        if "}" in tag:
            return tag.rsplit("}", 1)[-1]
        return tag

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
    def _parse_iso_datetime(value: str) -> datetime | None:
        if not value:
            return None
        try:
            parsed = datetime.fromisoformat(value)
        except ValueError:
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
        start = RSSFeedCollector._as_utc(window_start)
        end = RSSFeedCollector._as_utc(window_end)
        moment = RSSFeedCollector._as_utc(published_at)
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
