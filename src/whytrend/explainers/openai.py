"""OpenAI-backed explainer."""

from __future__ import annotations

from typing import Any

from whytrend.explainers.llm import LLMExplainer
from whytrend.llm.openai import OpenAIProvider


class OpenAIExplainer(LLMExplainer):
    """Explain anomalies with the OpenAI API."""

    def __init__(
        self,
        *,
        model: str = "gpt-4o-mini",
        api_key: str | None = None,
        base_url: str | None = None,
        client: Any | None = None,
    ) -> None:
        super().__init__(
            OpenAIProvider(
                model=model,
                api_key=api_key,
                base_url=base_url,
                client=client,
            ),
            name="openai",
        )
