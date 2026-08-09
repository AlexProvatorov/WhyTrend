"""OpenRouter-backed explainer."""

from __future__ import annotations

from typing import Any

from whytrend.explainers.llm import LLMExplainer
from whytrend.llm.openrouter import DEFAULT_BASE_URL, OpenRouterProvider


class OpenRouterExplainer(LLMExplainer):
    """Explain anomalies via OpenRouter (many models, one API)."""

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
        super().__init__(
            OpenRouterProvider(
                model=model,
                api_key=api_key,
                base_url=base_url,
                site_url=site_url,
                app_title=app_title,
                client=client,
            ),
            name="openrouter",
        )
