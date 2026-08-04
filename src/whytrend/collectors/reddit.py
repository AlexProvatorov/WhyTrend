"""Collect context from Reddit posts and comments."""

from __future__ import annotations

import os
import time
from datetime import UTC, datetime
from typing import Any

import httpx

from whytrend.collectors._http import DEFAULT_USER_AGENT, get_http_client
from whytrend.collectors._utils import evidence_from_fields, unix_to_datetime
from whytrend.core.models import Event, Evidence
from whytrend.core.protocols import BaseCollector

REDDIT_TOKEN_URL = "https://www.reddit.com/api/v1/access_token"
REDDIT_OAUTH_BASE = "https://oauth.reddit.com"


class RedditCollector(BaseCollector):
    """Search Reddit posts/comments around an event time window.

    Requires Reddit API application credentials (script/web app type with
    client credentials). Pass ``client_id`` / ``client_secret`` or set
    ``REDDIT_CLIENT_ID`` and ``REDDIT_CLIENT_SECRET`` environment variables.

    Create credentials at https://www.reddit.com/prefs/apps
    """

    def __init__(
        self,
        *,
        client_id: str | None = None,
        client_secret: str | None = None,
        access_token: str | None = None,
        timeout: float = 10.0,
        max_results: int = 10,
        subreddits: list[str] | None = None,
        user_agent: str = DEFAULT_USER_AGENT,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        if max_results < 1:
            msg = "max_results must be >= 1"
            raise ValueError(msg)
        if not user_agent.strip():
            msg = "user_agent cannot be empty"
            raise ValueError(msg)

        cleaned_subreddits = [item.strip() for item in (subreddits or []) if item.strip()]
        if subreddits is not None and not cleaned_subreddits:
            msg = "subreddits cannot be empty when provided"
            raise ValueError(msg)

        self._client_id = (client_id or os.getenv("REDDIT_CLIENT_ID") or "").strip() or None
        self._client_secret = (
            client_secret or os.getenv("REDDIT_CLIENT_SECRET") or ""
        ).strip() or None
        self._access_token = (access_token or "").strip() or None
        self._timeout = timeout
        self._max_results = max_results
        self._subreddits = cleaned_subreddits
        self._user_agent = user_agent.strip()
        self._client = client
        self._token_expires_at: float | None = None

    @property
    def name(self) -> str:
        return "reddit"

    async def collect(self, event: Event) -> list[Evidence]:
        async with get_http_client(timeout=self._timeout, client=self._client) as client:
            token = await self._resolve_access_token(client)
            children = await self._search(client, token=token, keyword=event.keyword)

        return self._parse_children(
            children,
            window_start=event.window_start,
            window_end=event.window_end,
        )

    async def _resolve_access_token(self, client: httpx.AsyncClient) -> str:
        if self._access_token is not None and (
            self._token_expires_at is None or time.monotonic() < self._token_expires_at
        ):
            return self._access_token

        if not self._client_id or not self._client_secret:
            msg = (
                "RedditCollector requires credentials. Pass client_id and "
                "client_secret, or set REDDIT_CLIENT_ID and REDDIT_CLIENT_SECRET."
            )
            raise ValueError(msg)

        response = await client.post(
            REDDIT_TOKEN_URL,
            data={"grant_type": "client_credentials"},
            auth=(self._client_id, self._client_secret),
            headers={"User-Agent": self._user_agent},
        )
        response.raise_for_status()
        payload = response.json()
        if not isinstance(payload, dict):
            msg = "Reddit token endpoint returned a non-object payload"
            raise TypeError(msg)

        token = str(payload.get("access_token") or "").strip()
        if not token:
            msg = "Reddit token endpoint did not return access_token"
            raise ValueError(msg)

        self._access_token = token
        expires_in = payload.get("expires_in")
        if isinstance(expires_in, (int, float)) and expires_in > 0:
            # Refresh a bit early to avoid racing the exact expiry.
            self._token_expires_at = time.monotonic() + float(expires_in) - 30.0
        else:
            self._token_expires_at = None
        return token

    async def _search(
        self,
        client: httpx.AsyncClient,
        *,
        token: str,
        keyword: str,
    ) -> list[Any]:
        headers = {
            "Authorization": f"Bearer {token}",
            "User-Agent": self._user_agent,
        }
        # Fetch extra rows so time-window filtering can still fill max_results.
        limit = min(100, self._max_results * 3)

        if not self._subreddits:
            payload = await self._get_search(
                client,
                url=f"{REDDIT_OAUTH_BASE}/search",
                headers=headers,
                params={
                    "q": keyword,
                    "sort": "new",
                    "limit": limit,
                    "restrict_sr": "false",
                    "raw_json": 1,
                },
            )
            return self._children_from_payload(payload)

        children: list[Any] = []
        seen_ids: set[str] = set()
        for subreddit in self._subreddits:
            payload = await self._get_search(
                client,
                url=f"{REDDIT_OAUTH_BASE}/r/{subreddit}/search",
                headers=headers,
                params={
                    "q": keyword,
                    "sort": "new",
                    "limit": limit,
                    "restrict_sr": "true",
                    "raw_json": 1,
                },
            )
            for child in self._children_from_payload(payload):
                if not isinstance(child, dict):
                    continue
                data = child.get("data")
                post_id = ""
                if isinstance(data, dict):
                    post_id = str(data.get("id") or "").strip()
                if post_id:
                    if post_id in seen_ids:
                        continue
                    seen_ids.add(post_id)
                children.append(child)
        return children

    @staticmethod
    async def _get_search(
        client: httpx.AsyncClient,
        *,
        url: str,
        headers: dict[str, str],
        params: dict[str, Any],
    ) -> dict[str, Any]:
        response = await client.get(url, headers=headers, params=params)
        response.raise_for_status()
        payload = response.json()
        if not isinstance(payload, dict):
            msg = f"expected JSON object from {url}"
            raise TypeError(msg)
        return payload

    @staticmethod
    def _children_from_payload(payload: dict[str, Any]) -> list[Any]:
        data = payload.get("data")
        if not isinstance(data, dict):
            return []
        children = data.get("children")
        if not isinstance(children, list):
            return []
        return children

    def _parse_children(
        self,
        children: Any,
        *,
        window_start: datetime,
        window_end: datetime,
    ) -> list[Evidence]:
        if not isinstance(children, list):
            return []

        evidences: list[Evidence] = []
        for child in children:
            evidence = self._parse_child(
                child,
                window_start=window_start,
                window_end=window_end,
            )
            if evidence is None:
                continue
            evidences.append(evidence)
            if len(evidences) >= self._max_results:
                break
        return evidences

    def _parse_child(
        self,
        child: Any,
        *,
        window_start: datetime,
        window_end: datetime,
    ) -> Evidence | None:
        if not isinstance(child, dict):
            return None

        kind = str(child.get("kind") or "").strip()
        data = child.get("data")
        if not isinstance(data, dict):
            return None

        created_utc = data.get("created_utc")
        published_at = None
        if created_utc is not None:
            try:
                published_at = unix_to_datetime(int(float(created_utc)))
            except (TypeError, ValueError):
                published_at = None

        if published_at is None or not self._in_window(
            published_at,
            window_start=window_start,
            window_end=window_end,
        ):
            return None

        title, snippet, url = self._extract_fields(kind=kind, data=data)
        if not title or not url:
            return None

        subreddit = str(data.get("subreddit") or "").strip()
        post_id = str(data.get("id") or "").strip()
        metadata: dict[str, Any] = {"kind": kind or None}
        if subreddit:
            metadata["subreddit"] = subreddit
        if post_id:
            metadata["id"] = post_id
        score = data.get("score")
        if isinstance(score, (int, float)):
            metadata["score"] = int(score)

        return evidence_from_fields(
            title=title,
            url=url,
            snippet=snippet[:500],
            source_name=self.name,
            published_at=published_at,
            metadata=metadata,
        )

    @staticmethod
    def _extract_fields(*, kind: str, data: dict[str, Any]) -> tuple[str, str, str]:
        permalink = str(data.get("permalink") or "").strip()
        url = ""
        if permalink:
            url = (
                permalink if permalink.startswith("http") else f"https://www.reddit.com{permalink}"
            )

        if kind == "t1":
            body = str(data.get("body") or "").strip()
            link_title = str(data.get("link_title") or "").strip()
            title = link_title or (body[:120] if body else "")
            return title, body, url

        title = str(data.get("title") or "").strip()
        snippet = str(data.get("selftext") or "").strip()
        if not url:
            url = str(data.get("url") or "").strip()
        return title, snippet, url

    @staticmethod
    def _in_window(
        published_at: datetime,
        *,
        window_start: datetime,
        window_end: datetime,
    ) -> bool:
        start = RedditCollector._as_utc(window_start)
        end = RedditCollector._as_utc(window_end)
        moment = RedditCollector._as_utc(published_at)
        return start <= moment <= end

    @staticmethod
    def _as_utc(moment: datetime) -> datetime:
        if moment.tzinfo is None:
            return moment.replace(tzinfo=UTC)
        return moment.astimezone(UTC)
