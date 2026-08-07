import json
from datetime import UTC, datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import pytest

from whytrend.core import AnomalyType, Event, Evidence
from whytrend.explainers import (
    AnthropicExplainer,
    DeepSeekExplainer,
    GeminiExplainer,
    LLMExplainer,
    OllamaExplainer,
    OpenAIExplainer,
)
from whytrend.llm import (
    AnthropicProvider,
    DeepSeekProvider,
    GeminiProvider,
    MockLLMProvider,
    OllamaProvider,
    OpenAIProvider,
)
from whytrend.llm._parsing import explanation_from_payload, fallback_explanation, parse_llm_content


@pytest.fixture
def llm_event() -> Event:
    return Event(
        anomaly_type=AnomalyType.SPIKE,
        timestamp=datetime(2026, 1, 4, tzinfo=UTC),
        window_start=datetime(2026, 1, 1, tzinfo=UTC),
        window_end=datetime(2026, 1, 7, tzinfo=UTC),
        keyword="Python",
        series_name="python_interest",
        value=610.0,
        detection_score=0.95,
    )


@pytest.fixture
def llm_evidences() -> list[Evidence]:
    return [
        Evidence(
            title="Python 3.13 released",
            url="https://example.com/python-3-13",
            snippet="Python 3.13 brings performance improvements.",
            source_name="hacker_news",
            relevance_score=0.92,
        )
    ]


def test_parse_llm_content_builds_explanation(llm_event: Event) -> None:
    payload = {
        "summary": "Interest in Python spiked due to the Python 3.13 release.",
        "confidence": 0.91,
        "causes": [
            {
                "source": "hacker_news",
                "score": 0.92,
                "url": "https://example.com/python-3-13",
                "title": "Python 3.13 released",
                "summary": "Release announcement.",
            }
        ],
    }

    explanation = parse_llm_content(llm_event, json.dumps(payload))

    assert explanation.summary.startswith("Interest in Python")
    assert explanation.confidence == 0.91
    assert explanation.event_id == llm_event.id
    assert explanation.causes[0].source == "hacker_news"


def test_fallback_explanation_without_evidence(llm_event: Event) -> None:
    explanation = fallback_explanation(llm_event, [])

    assert "no supporting external evidence" in explanation.summary
    assert explanation.causes == []


@pytest.mark.asyncio
async def test_mock_llm_provider_returns_explanation(llm_event: Event, llm_evidences) -> None:
    explanation = await MockLLMProvider().explain(llm_event, llm_evidences)

    assert "Python 3.13 released" in explanation.summary
    assert explanation.causes[0].url.endswith("python-3-13")


@pytest.mark.asyncio
async def test_openai_provider_parses_json_response(llm_event: Event, llm_evidences) -> None:
    payload = {
        "summary": "Interest in 'Python' spiked on January 4 due to Python 3.13.",
        "confidence": 0.94,
        "causes": [
            {
                "source": "hacker_news",
                "score": 0.92,
                "url": "https://example.com/python-3-13",
                "title": "Python 3.13 released",
                "summary": "Release announcement.",
            }
        ],
    }
    fake_client = SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(
                create=AsyncMock(
                    return_value=SimpleNamespace(
                        choices=[
                            SimpleNamespace(message=SimpleNamespace(content=json.dumps(payload)))
                        ]
                    )
                )
            )
        )
    )

    provider = OpenAIProvider(model="gpt-4o-mini", client=fake_client)
    explanation = await provider.explain(llm_event, llm_evidences)

    assert explanation.summary.startswith("Interest in 'Python'")
    assert explanation.metadata["provider"] == "openai"
    assert explanation.metadata["model"] == "gpt-4o-mini"
    fake_client.chat.completions.create.assert_awaited_once()


@pytest.mark.asyncio
async def test_ollama_provider_parses_json_response(llm_event: Event, llm_evidences) -> None:
    payload = {
        "summary": "Local model links the spike to Python 3.13.",
        "confidence": 0.88,
        "causes": [
            {
                "source": "hacker_news",
                "score": 0.9,
                "url": "https://example.com/python-3-13",
                "title": "Python 3.13 released",
                "summary": "Release announcement.",
            }
        ],
    }

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/chat"
        body = json.loads(request.content.decode())
        assert body["model"] == "llama3.2"
        assert body["format"] == "json"
        return httpx.Response(
            200,
            json={"message": {"role": "assistant", "content": json.dumps(payload)}},
        )

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    provider = OllamaProvider(model="llama3.2", client=client)
    explanation = await provider.explain(llm_event, llm_evidences)

    assert "Python 3.13" in explanation.summary
    assert explanation.metadata["provider"] == "ollama"


@pytest.mark.asyncio
async def test_openai_explainer_wraps_provider(llm_event: Event, llm_evidences) -> None:
    provider = MockLLMProvider(summary_prefix="OpenAI-style")
    explainer = LLMExplainer(provider, name="openai")

    explanation = await explainer.explain(llm_event, llm_evidences)

    assert explainer.name == "openai"
    assert explanation.summary.startswith("OpenAI-style")


def test_openai_explainer_accepts_injected_client() -> None:
    explainer = OpenAIExplainer(client=SimpleNamespace())
    assert explainer.name == "openai"
    assert isinstance(explainer.provider, OpenAIProvider)


def test_ollama_explainer_defaults() -> None:
    explainer = OllamaExplainer()
    assert explainer.name == "ollama"
    assert isinstance(explainer.provider, OllamaProvider)


@pytest.mark.asyncio
async def test_anthropic_provider_parses_json_response(llm_event: Event, llm_evidences) -> None:
    payload = {
        "summary": "Claude links the spike to Python 3.13.",
        "confidence": 0.93,
        "causes": [
            {
                "source": "hacker_news",
                "score": 0.92,
                "url": "https://example.com/python-3-13",
                "title": "Python 3.13 released",
                "summary": "Release announcement.",
            }
        ],
    }
    fake_client = SimpleNamespace(
        messages=SimpleNamespace(
            create=AsyncMock(
                return_value=SimpleNamespace(content=[SimpleNamespace(text=json.dumps(payload))])
            )
        )
    )

    provider = AnthropicProvider(model="claude-sonnet-4-20250514", client=fake_client)
    explanation = await provider.explain(llm_event, llm_evidences)

    assert "Python 3.13" in explanation.summary
    assert explanation.metadata["provider"] == "anthropic"
    assert explanation.metadata["model"] == "claude-sonnet-4-20250514"
    fake_client.messages.create.assert_awaited_once()


@pytest.mark.asyncio
async def test_gemini_provider_parses_json_response(llm_event: Event, llm_evidences) -> None:
    payload = {
        "summary": "Gemini links the spike to Python 3.13.",
        "confidence": 0.9,
        "causes": [
            {
                "source": "hacker_news",
                "score": 0.9,
                "url": "https://example.com/python-3-13",
                "title": "Python 3.13 released",
                "summary": "Release announcement.",
            }
        ],
    }

    def handler(request: httpx.Request) -> httpx.Response:
        assert "gemini-2.0-flash:generateContent" in str(request.url)
        assert request.url.params.get("key") == "test-gemini-key"
        body = json.loads(request.content.decode())
        assert body["generationConfig"]["responseMimeType"] == "application/json"
        return httpx.Response(
            200,
            json={"candidates": [{"content": {"parts": [{"text": json.dumps(payload)}]}}]},
        )

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    provider = GeminiProvider(api_key="test-gemini-key", client=client)
    explanation = await provider.explain(llm_event, llm_evidences)

    assert "Python 3.13" in explanation.summary
    assert explanation.metadata["provider"] == "gemini"
    assert explanation.metadata["model"] == "gemini-2.0-flash"


@pytest.mark.asyncio
async def test_deepseek_provider_parses_json_response(llm_event: Event, llm_evidences) -> None:
    payload = {
        "summary": "DeepSeek links the spike to Python 3.13.",
        "confidence": 0.89,
        "causes": [
            {
                "source": "hacker_news",
                "score": 0.91,
                "url": "https://example.com/python-3-13",
                "title": "Python 3.13 released",
                "summary": "Release announcement.",
            }
        ],
    }
    fake_client = SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(
                create=AsyncMock(
                    return_value=SimpleNamespace(
                        choices=[
                            SimpleNamespace(message=SimpleNamespace(content=json.dumps(payload)))
                        ]
                    )
                )
            )
        )
    )

    provider = DeepSeekProvider(model="deepseek-chat", client=fake_client)
    explanation = await provider.explain(llm_event, llm_evidences)

    assert "Python 3.13" in explanation.summary
    assert explanation.metadata["provider"] == "deepseek"
    fake_client.chat.completions.create.assert_awaited_once()


def test_anthropic_provider_requires_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    provider = AnthropicProvider()
    with pytest.raises(ValueError, match="ANTHROPIC_API_KEY"):
        provider._resolve_api_key()


def test_gemini_provider_requires_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    provider = GeminiProvider()
    with pytest.raises(ValueError, match="GEMINI_API_KEY"):
        provider._resolve_api_key()


def test_deepseek_provider_requires_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    provider = DeepSeekProvider()
    with pytest.raises(ValueError, match="DEEPSEEK_API_KEY"):
        provider._resolve_api_key()


def test_new_explainers_accept_injected_clients() -> None:
    assert AnthropicExplainer(client=SimpleNamespace()).name == "anthropic"
    assert isinstance(AnthropicExplainer(client=SimpleNamespace()).provider, AnthropicProvider)
    assert GeminiExplainer(api_key="x").name == "gemini"
    assert isinstance(GeminiExplainer(api_key="x").provider, GeminiProvider)
    assert DeepSeekExplainer(client=SimpleNamespace()).name == "deepseek"
    assert isinstance(DeepSeekExplainer(client=SimpleNamespace()).provider, DeepSeekProvider)


def test_explanation_from_payload_rejects_invalid_confidence(llm_event: Event) -> None:
    with pytest.raises(ValueError, match="expected explanation schema"):
        explanation_from_payload(
            llm_event,
            {
                "summary": "Bad payload",
                "confidence": 1.5,
                "causes": [],
            },
        )
