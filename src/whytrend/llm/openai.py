"""OpenAI chat-completions provider."""

from __future__ import annotations

from typing import Any

from whytrend.core.models import Event, Evidence, Explanation
from whytrend.core.protocols import BaseLLMProvider
from whytrend.llm._parsing import parse_llm_content
from whytrend.llm.prompts import SYSTEM_PROMPT, build_user_prompt


class OpenAIProvider(BaseLLMProvider):
    """Generate explanations with the OpenAI Chat Completions API."""

    def __init__(
        self,
        *,
        model: str = "gpt-4o-mini",
        api_key: str | None = None,
        base_url: str | None = None,
        client: Any | None = None,
    ) -> None:
        if not model.strip():
            msg = "model cannot be empty"
            raise ValueError(msg)

        self._model = model
        self._api_key = api_key
        self._base_url = base_url
        self._client = client

    @property
    def name(self) -> str:
        return "openai"

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
            msg = "OpenAI returned an empty response"
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
            msg = "OpenAIProvider requires openai; install with: pip install 'whytrend[openai]'"
            raise ImportError(msg) from exc

        kwargs: dict[str, Any] = {}
        if self._api_key is not None:
            kwargs["api_key"] = self._api_key
        if self._base_url is not None:
            kwargs["base_url"] = self._base_url
        return AsyncOpenAI(**kwargs)
