"""Azure OpenAI-backed explainer."""

from __future__ import annotations

from typing import Any

from whytrend.explainers.llm import LLMExplainer
from whytrend.llm.azure import DEFAULT_API_VERSION, AzureOpenAIProvider


class AzureOpenAIExplainer(LLMExplainer):
    """Explain anomalies with Azure OpenAI."""

    def __init__(
        self,
        *,
        deployment: str,
        azure_endpoint: str | None = None,
        api_key: str | None = None,
        api_version: str = DEFAULT_API_VERSION,
        client: Any | None = None,
    ) -> None:
        super().__init__(
            AzureOpenAIProvider(
                deployment=deployment,
                azure_endpoint=azure_endpoint,
                api_key=api_key,
                api_version=api_version,
                client=client,
            ),
            name="azure_openai",
        )
