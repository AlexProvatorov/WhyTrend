"""Anthropic-backed explainer."""

from __future__ import annotations

from typing import Any

from whytrend.explainers.llm import LLMExplainer
from whytrend.llm.anthropic import AnthropicProvider


class AnthropicExplainer(LLMExplainer):
    """Explain anomalies with Anthropic Claude."""

    def __init__(
        self,
        *,
        model: str = "claude-sonnet-4-20250514",
        api_key: str | None = None,
        max_tokens: int = 4096,
        client: Any | None = None,
    ) -> None:
        super().__init__(
            AnthropicProvider(
                model=model,
                api_key=api_key,
                max_tokens=max_tokens,
                client=client,
            ),
            name="anthropic",
        )
