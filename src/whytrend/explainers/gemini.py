"""Google Gemini-backed explainer."""

from __future__ import annotations

import httpx

from whytrend.explainers.llm import LLMExplainer
from whytrend.llm.gemini import DEFAULT_BASE_URL, GeminiProvider


class GeminiExplainer(LLMExplainer):
    """Explain anomalies with Google Gemini."""

    def __init__(
        self,
        *,
        model: str = "gemini-2.0-flash",
        api_key: str | None = None,
        base_url: str = DEFAULT_BASE_URL,
        timeout: float = 60.0,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        super().__init__(
            GeminiProvider(
                model=model,
                api_key=api_key,
                base_url=base_url,
                timeout=timeout,
                client=client,
            ),
            name="gemini",
        )
