"""HTTP helpers for collector implementations."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

import httpx


@asynccontextmanager
async def get_http_client(
    *,
    timeout: float,
    client: httpx.AsyncClient | None = None,
) -> AsyncIterator[httpx.AsyncClient]:
    """Yield an HTTP client, creating one when not provided."""
    if client is not None:
        yield client
        return

    async with httpx.AsyncClient(timeout=timeout) as managed_client:
        yield managed_client


async def get_json(
    client: httpx.AsyncClient,
    url: str,
    *,
    params: dict[str, Any] | None = None,
) -> dict[str, Any]:
    response = await client.get(url, params=params)
    response.raise_for_status()
    payload = response.json()
    if not isinstance(payload, dict):
        msg = f"expected JSON object from {url}"
        raise TypeError(msg)
    return payload
