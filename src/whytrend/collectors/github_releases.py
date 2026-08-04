"""Collect context from GitHub Releases."""

from __future__ import annotations

import os
from datetime import UTC, datetime
from typing import Any

import httpx

from whytrend.collectors._http import DEFAULT_USER_AGENT, get_http_client
from whytrend.collectors._utils import evidence_from_fields, keyword_matches
from whytrend.core.models import Event, Evidence
from whytrend.core.protocols import BaseCollector

GITHUB_API_BASE = "https://api.github.com"


class GitHubReleasesCollector(BaseCollector):
    """Find GitHub release announcements related to an event keyword.

    Searches repositories matching ``event.keyword``, then fetches their
    releases and keeps those published within ``[window_start, window_end]``.

    Works without authentication for light use. Set ``token`` or
    ``GITHUB_TOKEN`` for higher GitHub API rate limits.
    """

    def __init__(
        self,
        *,
        token: str | None = None,
        timeout: float = 10.0,
        max_results: int = 10,
        max_repos: int = 5,
        repos: list[str] | None = None,
        user_agent: str = DEFAULT_USER_AGENT,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        if max_results < 1:
            msg = "max_results must be >= 1"
            raise ValueError(msg)
        if max_repos < 1:
            msg = "max_repos must be >= 1"
            raise ValueError(msg)
        if not user_agent.strip():
            msg = "user_agent cannot be empty"
            raise ValueError(msg)

        cleaned_repos = [item.strip() for item in (repos or []) if item.strip()]
        if repos is not None and not cleaned_repos:
            msg = "repos cannot be empty when provided"
            raise ValueError(msg)
        for repo in cleaned_repos:
            if "/" not in repo:
                msg = f"repo must be 'owner/name', got '{repo}'"
                raise ValueError(msg)

        self._token = (token or os.getenv("GITHUB_TOKEN") or "").strip() or None
        self._timeout = timeout
        self._max_results = max_results
        self._max_repos = max_repos
        self._repos = cleaned_repos
        self._user_agent = user_agent.strip()
        self._client = client

    @property
    def name(self) -> str:
        return "github_releases"

    async def collect(self, event: Event) -> list[Evidence]:
        async with get_http_client(timeout=self._timeout, client=self._client) as client:
            try:
                repo_full_names = await self._resolve_repos(client, keyword=event.keyword)
                releases = await self._fetch_releases(client, repo_full_names)
            except httpx.HTTPError:
                return []

        return self._parse_releases(
            releases,
            keyword=event.keyword,
            window_start=event.window_start,
            window_end=event.window_end,
        )

    async def _resolve_repos(self, client: httpx.AsyncClient, *, keyword: str) -> list[str]:
        if self._repos:
            return self._repos[: self._max_repos]

        payload = await self._get_json(
            client,
            f"{GITHUB_API_BASE}/search/repositories",
            params={
                "q": f"{keyword} in:name",
                "sort": "updated",
                "order": "desc",
                "per_page": self._max_repos,
            },
        )
        items = payload.get("items")
        if not isinstance(items, list):
            return []

        names: list[str] = []
        for item in items:
            if not isinstance(item, dict):
                continue
            full_name = str(item.get("full_name") or "").strip()
            if full_name:
                names.append(full_name)
        return names

    async def _fetch_releases(
        self,
        client: httpx.AsyncClient,
        repo_full_names: list[str],
    ) -> list[dict[str, Any]]:
        releases: list[dict[str, Any]] = []
        per_page = min(30, max(self._max_results * 2, self._max_results))

        for full_name in repo_full_names:
            response = await client.get(
                f"{GITHUB_API_BASE}/repos/{full_name}/releases",
                headers=self._headers(),
                params={"per_page": per_page},
            )
            if response.is_error:
                continue
            response.raise_for_status()
            payload = response.json()
            if not isinstance(payload, list):
                continue

            for item in payload:
                if not isinstance(item, dict):
                    continue
                enriched = dict(item)
                enriched["_repo_full_name"] = full_name
                releases.append(enriched)

        return releases

    async def _get_json(
        self,
        client: httpx.AsyncClient,
        url: str,
        *,
        params: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        response = await client.get(url, headers=self._headers(), params=params)
        response.raise_for_status()
        payload = response.json()
        if not isinstance(payload, dict):
            msg = f"expected JSON object from {url}"
            raise TypeError(msg)
        return payload

    def _headers(self) -> dict[str, str]:
        headers = {
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
            "User-Agent": self._user_agent,
        }
        if self._token is not None:
            headers["Authorization"] = f"Bearer {self._token}"
        return headers

    def _parse_releases(
        self,
        releases: Any,
        *,
        keyword: str,
        window_start: datetime,
        window_end: datetime,
    ) -> list[Evidence]:
        if not isinstance(releases, list):
            return []

        keyword_lower = keyword.casefold()
        evidences: list[Evidence] = []
        for release in releases:
            evidence = self._parse_release(
                release,
                keyword_lower=keyword_lower,
                window_start=window_start,
                window_end=window_end,
            )
            if evidence is None:
                continue
            evidences.append(evidence)
            if len(evidences) >= self._max_results:
                break
        return evidences

    def _parse_release(
        self,
        release: Any,
        *,
        keyword_lower: str,
        window_start: datetime,
        window_end: datetime,
    ) -> Evidence | None:
        if not isinstance(release, dict):
            return None

        published_at = self._parse_iso_datetime(str(release.get("published_at") or ""))
        if published_at is None:
            return None
        if not self._in_window(
            published_at,
            window_start=window_start,
            window_end=window_end,
        ):
            return None

        tag_name = str(release.get("tag_name") or "").strip()
        name = str(release.get("name") or "").strip()
        body = str(release.get("body") or "").strip()
        html_url = str(release.get("html_url") or "").strip()
        repo_full_name = str(release.get("_repo_full_name") or "").strip()

        if not html_url:
            return None

        haystack = f"{repo_full_name} {tag_name} {name} {body}"
        if keyword_lower and not keyword_matches(haystack, keyword_lower):
            return None

        title = name or (f"{repo_full_name} {tag_name}".strip() if tag_name else repo_full_name)
        if not title:
            return None

        metadata: dict[str, Any] = {}
        if repo_full_name:
            metadata["repo"] = repo_full_name
        if tag_name:
            metadata["tag_name"] = tag_name
        release_id = release.get("id")
        if release_id is not None:
            metadata["id"] = release_id

        return evidence_from_fields(
            title=title,
            url=html_url,
            snippet=body[:500],
            source_name=self.name,
            published_at=published_at,
            metadata=metadata,
        )

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
        start = GitHubReleasesCollector._as_utc(window_start)
        end = GitHubReleasesCollector._as_utc(window_end)
        moment = GitHubReleasesCollector._as_utc(published_at)
        return start <= moment <= end

    @staticmethod
    def _as_utc(moment: datetime) -> datetime:
        if moment.tzinfo is None:
            return moment.replace(tzinfo=UTC)
        return moment.astimezone(UTC)
