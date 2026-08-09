"""DeepSeek-backed explainer."""

from __future__ import annotations

from typing import Any

from whytrend.explainers.llm import LLMExplainer
from whytrend.llm.deepseek import DEFAULT_BASE_URL, DeepSeekProvider


class DeepSeekExplainer(LLMExplainer):
    """Explain anomalies with DeepSeek (OpenAI-compatible API)."""

    def __init__(
        self,
        *,
        model: str = "deepseek-chat",
        api_key: str | None = None,
        base_url: str = DEFAULT_BASE_URL,
        client: Any | None = None,
    ) -> None:
        super().__init__(
            DeepSeekProvider(
                model=model,
                api_key=api_key,
                base_url=base_url,
                client=client,
            ),
            name="deepseek",
        )
