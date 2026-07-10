"""Ollama-backed local explainer."""

from __future__ import annotations

import httpx

from whytrend.explainers.llm import LLMExplainer
from whytrend.llm.ollama import OllamaProvider


class OllamaExplainer(LLMExplainer):
    """Explain anomalies with a local Ollama model."""

    def __init__(
        self,
        *,
        model: str = "llama3.2",
        base_url: str = "http://localhost:11434",
        timeout: float = 60.0,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        super().__init__(
            OllamaProvider(
                model=model,
                base_url=base_url,
                timeout=timeout,
                client=client,
            ),
            name="ollama",
        )
