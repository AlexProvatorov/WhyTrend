"""Ollama local LLM provider."""

from __future__ import annotations

from typing import Any

import httpx

from whytrend.collectors._http import get_http_client
from whytrend.core.models import Event, Evidence, Explanation
from whytrend.core.protocols import BaseLLMProvider
from whytrend.llm._parsing import parse_llm_content
from whytrend.llm.prompts import SYSTEM_PROMPT, build_user_prompt


class OllamaProvider(BaseLLMProvider):
    """Generate explanations with a local Ollama server."""

    def __init__(
        self,
        *,
        model: str = "llama3.2",
        base_url: str = "http://localhost:11434",
        timeout: float = 60.0,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        if not model.strip():
            msg = "model cannot be empty"
            raise ValueError(msg)
        if not base_url.strip():
            msg = "base_url cannot be empty"
            raise ValueError(msg)

        self._model = model
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout
        self._client = client

    @property
    def name(self) -> str:
        return "ollama"

    @property
    def chat_url(self) -> str:
        return f"{self._base_url}/api/chat"

    async def explain(self, event: Event, evidences: list[Evidence]) -> Explanation:
        payload = {
            "model": self._model,
            "stream": False,
            "format": "json",
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": build_user_prompt(event, evidences)},
            ],
        }

        async with get_http_client(timeout=self._timeout, client=self._client) as client:
            response_payload = await self._post_json(client, self.chat_url, payload)

        message = response_payload.get("message")
        if not isinstance(message, dict):
            msg = "Ollama returned an unexpected response payload"
            raise ValueError(msg)

        content = message.get("content")
        if not isinstance(content, str) or not content.strip():
            msg = "Ollama returned an empty response"
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

    async def _post_json(
        self,
        client: httpx.AsyncClient,
        url: str,
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        response = await client.post(url, json=payload)
        response.raise_for_status()
        body = response.json()
        if not isinstance(body, dict):
            msg = f"expected JSON object from {url}"
            raise TypeError(msg)
        return body
