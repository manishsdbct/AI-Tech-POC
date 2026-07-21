from .anthropic_provider import AnthropicAdapter
from .openai_provider import OpenAIAdapter
from .base import ProviderAdapter, ProviderResult

ADAPTERS: dict[str, ProviderAdapter] = {
    "openai": OpenAIAdapter(),
    "anthropic": AnthropicAdapter(),
}

__all__ = ["ADAPTERS", "ProviderAdapter", "ProviderResult"]
