"""Common interface every provider adapter must implement.

Callers of the router only ever see this shape, regardless of which vendor
SDK sits underneath — that's what makes swapping providers a config change
instead of a rewrite.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class ProviderResult:
    content: str
    prompt_tokens: int
    completion_tokens: int


class ProviderAdapter(ABC):
    name: str

    @abstractmethod
    async def complete(self, model: str, messages: list[dict]) -> ProviderResult:
        """Call the underlying provider and return a normalized result."""
