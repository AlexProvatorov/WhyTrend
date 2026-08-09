"""Google Gemini LLM provider (REST via httpx)."""

from __future__ import annotations

import os
from typing import Any

import httpx

from whytrend.collectors._http import get_http_client
from whytrend.core.models import Event, Evidence, Explanation
from whytrend.core.protocols import BaseLLMProvider
from whytrend.llm._parsing import parse_llm_content
from whytrend.llm.prompts import SYSTEM_PROMPT, build_user_prompt

DEFAULT_BASE_URL = "https://generativelanguage.googleapis.com/v1beta"


class GeminiProvider(BaseLLMProvider):
    """Generate explanations with the Gemini generateContent API."""

    def __init__(
        self,
        *,
        model: str = "gemini-2.0-flash",
        api_key: str | None = None,
        base_url: str = DEFAULT_BASE_URL,
        timeout: float = 60.0,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        if not model.strip():
            msg = "model cannot be empty"
            raise ValueError(msg)
        if not base_url.strip():
            msg = "base_url cannot be empty"
            raise ValueError(msg)

        self._model = model
        self._api_key = api_key
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout
        self._client = client

    @property
    def name(self) -> str:
        return "gemini"

    @property
    def generate_url(self) -> str:
        return f"{self._base_url}/models/{self._model}:generateContent"

    async def explain(self, event: Event, evidences: list[Evidence]) -> Explanation:
        api_key = self._resolve_api_key()
        payload = {
            "system_instruction": {"parts": [{"text": SYSTEM_PROMPT}]},
            "contents": [
                {
                    "role": "user",
                    "parts": [{"text": build_user_prompt(event, evidences)}],
                }
            ],
            "generationConfig": {
                "temperature": 0.2,
                "responseMimeType": "application/json",
            },
        }

        async with get_http_client(timeout=self._timeout, client=self._client) as client:
            response = await client.post(
                self.generate_url,
                params={"key": api_key},
                json=payload,
            )
            response.raise_for_status()
            body = response.json()

        content = self._extract_text(body)
        if not content.strip():
            msg = "Gemini returned an empty response"
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

    def _resolve_api_key(self) -> str:
        key = (
            self._api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY") or ""
        ).strip()
        if not key:
            msg = "GeminiProvider requires api_key=... or GEMINI_API_KEY / GOOGLE_API_KEY"
            raise ValueError(msg)
        return key

    @staticmethod
    def _extract_text(body: Any) -> str:
        if not isinstance(body, dict):
            return ""
        candidates = body.get("candidates")
        if not isinstance(candidates, list) or not candidates:
            return ""
        first = candidates[0]
        if not isinstance(first, dict):
            return ""
        content = first.get("content")
        if not isinstance(content, dict):
            return ""
        parts = content.get("parts")
        if not isinstance(parts, list):
            return ""
        texts: list[str] = []
        for part in parts:
            if isinstance(part, dict) and isinstance(part.get("text"), str):
                texts.append(part["text"])
        return "".join(texts)
