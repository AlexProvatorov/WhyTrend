"""Azure OpenAI LLM provider."""

from __future__ import annotations

import os
from typing import Any

from whytrend.core.models import Event, Evidence, Explanation
from whytrend.core.protocols import BaseLLMProvider
from whytrend.llm._parsing import parse_llm_content
from whytrend.llm.prompts import SYSTEM_PROMPT, build_user_prompt

DEFAULT_API_VERSION = "2024-10-21"


class AzureOpenAIProvider(BaseLLMProvider):
    """Generate explanations with Azure OpenAI Chat Completions."""

    def __init__(
        self,
        *,
        deployment: str,
        azure_endpoint: str | None = None,
        api_key: str | None = None,
        api_version: str = DEFAULT_API_VERSION,
        client: Any | None = None,
    ) -> None:
        if not deployment.strip():
            msg = "AzureOpenAIProvider requires a non-empty deployment name"
            raise ValueError(msg)
        if not api_version.strip():
            msg = "api_version cannot be empty"
            raise ValueError(msg)

        self._deployment = deployment
        self._azure_endpoint = azure_endpoint
        self._api_key = api_key
        self._api_version = api_version
        self._client = client

    @property
    def name(self) -> str:
        return "azure_openai"

    async def explain(self, event: Event, evidences: list[Evidence]) -> Explanation:
        client = self._client or self._create_client()
        response = await client.chat.completions.create(
            model=self._deployment,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": build_user_prompt(event, evidences)},
            ],
            response_format={"type": "json_object"},
            temperature=0.2,
        )
        content = response.choices[0].message.content
        if not content:
            msg = "Azure OpenAI returned an empty response"
            raise ValueError(msg)

        explanation = parse_llm_content(event, content)
        return explanation.model_copy(
            update={
                "metadata": {
                    **explanation.metadata,
                    "provider": self.name,
                    "model": self._deployment,
                    "deployment": self._deployment,
                    "evidence_count": len(evidences),
                }
            }
        )

    def _create_client(self) -> Any:
        try:
            from openai import AsyncAzureOpenAI
        except ImportError as exc:
            msg = "AzureOpenAIProvider requires openai; install with: pip install 'whytrend[azure]'"
            raise ImportError(msg) from exc

        return AsyncAzureOpenAI(
            api_key=self._resolve_api_key(),
            azure_endpoint=self._resolve_endpoint(),
            api_version=self._api_version,
        )

    def _resolve_api_key(self) -> str:
        key = (self._api_key or os.getenv("AZURE_OPENAI_API_KEY") or "").strip()
        if not key:
            msg = "AzureOpenAIProvider requires api_key=... or AZURE_OPENAI_API_KEY"
            raise ValueError(msg)
        return key

    def _resolve_endpoint(self) -> str:
        endpoint = (self._azure_endpoint or os.getenv("AZURE_OPENAI_ENDPOINT") or "").strip()
        if not endpoint:
            msg = "AzureOpenAIProvider requires azure_endpoint=... or AZURE_OPENAI_ENDPOINT"
            raise ValueError(msg)
        return endpoint.rstrip("/")
