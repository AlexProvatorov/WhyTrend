"""DeepSeek LLM provider (OpenAI-compatible Chat Completions API)."""

from __future__ import annotations

import os
from typing import Any

from whytrend.core.models import Event, Evidence, Explanation
from whytrend.core.protocols import BaseLLMProvider
from whytrend.llm._parsing import parse_llm_content
from whytrend.llm.prompts import SYSTEM_PROMPT, build_user_prompt

DEFAULT_BASE_URL = "https://api.deepseek.com"


class DeepSeekProvider(BaseLLMProvider):
    """Generate explanations with DeepSeek's OpenAI-compatible API."""

    def __init__(
        self,
        *,
        model: str = "deepseek-chat",
        api_key: str | None = None,
        base_url: str = DEFAULT_BASE_URL,
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
        self._client = client

    @property
    def name(self) -> str:
        return "deepseek"

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
            msg = "DeepSeek returned an empty response"
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
            msg = "DeepSeekProvider requires openai; install with: pip install 'whytrend[deepseek]'"
            raise ImportError(msg) from exc

        return AsyncOpenAI(api_key=self._resolve_api_key(), base_url=self._base_url)

    def _resolve_api_key(self) -> str:
        key = (self._api_key or os.getenv("DEEPSEEK_API_KEY") or "").strip()
        if not key:
            msg = "DeepSeekProvider requires api_key=... or DEEPSEEK_API_KEY"
            raise ValueError(msg)
        return key
