"""OpenRouter LLM provider (OpenAI-compatible API)."""

from __future__ import annotations

import os
from typing import Any

from whytrend.core.models import Event, Evidence, Explanation
from whytrend.core.protocols import BaseLLMProvider
from whytrend.llm._parsing import parse_llm_content
from whytrend.llm.prompts import SYSTEM_PROMPT, build_user_prompt

DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"


class OpenRouterProvider(BaseLLMProvider):
    """Generate explanations via OpenRouter's OpenAI-compatible API."""

    def __init__(
        self,
        *,
        model: str = "openai/gpt-4o-mini",
        api_key: str | None = None,
        base_url: str = DEFAULT_BASE_URL,
        site_url: str | None = None,
        app_title: str | None = None,
        client: Any | None = None,
    ) -> None:
        if not model.strip():
            msg = "model cannot be empty"
            raise ValueError(msg)
        if not base_url.strip():
            msg = "base_url cannot be empty"
            raise ValueError(msg)

        self._model = model
        self._api_key = api_key
        self._base_url = base_url.rstrip("/")
        self._site_url = site_url
        self._app_title = app_title
        self._client = client

    @property
    def name(self) -> str:
        return "openrouter"

    async def explain(self, event: Event, evidences: list[Evidence]) -> Explanation:
        client = self._client or self._create_client()
        response = await client.chat.completions.create(
            model=self._model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": build_user_prompt(event, evidences)},
            ],
            response_format={"type": "json_object"},
            temperature=0.2,
        )
        content = response.choices[0].message.content
        if not content:
            msg = "OpenRouter returned an empty response"
            raise ValueError(msg)

        explanation = parse_llm_content(event, content)
        return explanation.model_copy(
            update={
                "metadata": {
                    **explanation.metadata,
                    "provider": self.name,
                    "model": self._model,
                    "evidence_count": len(evidences),
                }
            }
        )

    def _create_client(self) -> Any:
        try:
            from openai import AsyncOpenAI
        except ImportError as exc:
            msg = (
                "OpenRouterProvider requires openai; "
                "install with: pip install 'whytrend[openrouter]'"
            )
            raise ImportError(msg) from exc

        headers: dict[str, str] = {}
        if self._site_url:
            headers["HTTP-Referer"] = self._site_url
        if self._app_title:
            headers["X-Title"] = self._app_title

        kwargs: dict[str, Any] = {
            "api_key": self._resolve_api_key(),
            "base_url": self._base_url,
        }
        if headers:
            kwargs["default_headers"] = headers
        return AsyncOpenAI(**kwargs)

    def _resolve_api_key(self) -> str:
        key = (self._api_key or os.getenv("OPENROUTER_API_KEY") or "").strip()
        if not key:
            msg = "OpenRouterProvider requires api_key=... or OPENROUTER_API_KEY"
            raise ValueError(msg)
        return key
