"""LLM provider adapters."""

from whytrend.llm.anthropic import AnthropicProvider
from whytrend.llm.azure import AzureOpenAIProvider
from whytrend.llm.deepseek import DeepSeekProvider
from whytrend.llm.gemini import GeminiProvider
from whytrend.llm.mock import MockLLMProvider
from whytrend.llm.ollama import OllamaProvider
from whytrend.llm.openai import OpenAIProvider
from whytrend.llm.openrouter import OpenRouterProvider

__all__ = [
    "AnthropicProvider",
    "AzureOpenAIProvider",
    "DeepSeekProvider",
    "GeminiProvider",
    "MockLLMProvider",
    "OllamaProvider",
    "OpenAIProvider",
    "OpenRouterProvider",
]
