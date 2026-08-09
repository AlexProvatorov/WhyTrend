"""Anthropic Claude LLM provider."""

from __future__ import annotations

import os
from typing import Any

from whytrend.core.models import Event, Evidence, Explanation
from whytrend.core.protocols import BaseLLMProvider
from whytrend.llm._parsing import parse_llm_content
from whytrend.llm.prompts import SYSTEM_PROMPT, build_user_prompt


class AnthropicProvider(BaseLLMProvider):
    """Generate explanations with the Anthropic Messages API."""

    def __init__(
        self,
        *,
        model: str = "claude-sonnet-4-20250514",
        api_key: str | None = None,
        max_tokens: int = 4096,
        client: Any | None = None,
    ) -> None:
        if not model.strip():
            msg = "model cannot be empty"
            raise ValueError(msg)
        if max_tokens < 1:
            msg = "max_tokens must be >= 1"
            raise ValueError(msg)

        self._model = model
        self._api_key = api_key
        self._max_tokens = max_tokens
        self._client = client

    @property
    def name(self) -> str:
        return "anthropic"

    async def explain(self, event: Event, evidences: list[Evidence]) -> Explanation:
        client = self._client or self._create_client()
        response = await client.messages.create(
            model=self._model,
            max_tokens=self._max_tokens,
            system=SYSTEM_PROMPT,
            messages=[
                {"role": "user", "content": build_user_prompt(event, evidences)},
            ],
            temperature=0.2,
        )
        content = self._extract_text(response)
        if not content.strip():
            msg = "Anthropic returned an empty response"
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
            from anthropic import AsyncAnthropic
        except ImportError as exc:
            msg = (
                "AnthropicProvider requires anthropic; "
                "install with: pip install 'whytrend[anthropic]'"
            )
            raise ImportError(msg) from exc

        return AsyncAnthropic(api_key=self._resolve_api_key())

    def _resolve_api_key(self) -> str:
        key = (self._api_key or os.getenv("ANTHROPIC_API_KEY") or "").strip()
        if not key:
            msg = "AnthropicProvider requires api_key=... or ANTHROPIC_API_KEY"
            raise ValueError(msg)
        return key

    @staticmethod
    def _extract_text(response: Any) -> str:
        blocks = getattr(response, "content", None) or []
        parts: list[str] = []
        for block in blocks:
            text = getattr(block, "text", None)
            if isinstance(text, str):
                parts.append(text)
            elif isinstance(block, dict) and isinstance(block.get("text"), str):
                parts.append(block["text"])
        return "".join(parts)
