"""Explainer implementations built on LLM providers."""

from __future__ import annotations

from whytrend.core.models import Event, Evidence, Explanation
from whytrend.core.protocols import BaseExplainer, LLMProvider


class LLMExplainer(BaseExplainer):
    """Delegate explanation generation to an :class:`LLMProvider`."""

    def __init__(self, provider: LLMProvider, *, name: str | None = None) -> None:
        self._provider = provider
        self._name = name or provider.name

    @property
    def name(self) -> str:
        return self._name

    @property
    def provider(self) -> LLMProvider:
        return self._provider

    async def explain(self, event: Event, evidences: list[Evidence]) -> Explanation:
        return await self._provider.explain(event, evidences)
