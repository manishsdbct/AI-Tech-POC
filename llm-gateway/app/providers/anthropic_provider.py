"""Anthropic adapter. Requires ANTHROPIC_API_KEY; falls back to a mock
response when the SDK/key isn't configured, so the gateway is runnable
without live credentials during local dev.
"""
import os

from .base import ProviderAdapter, ProviderResult


class AnthropicAdapter(ProviderAdapter):
    name = "anthropic"

    async def complete(self, model: str, messages: list[dict]) -> ProviderResult:
        try:
            from anthropic import AsyncAnthropic
        except ImportError:
            return self._mock(messages)

        if not os.environ.get("ANTHROPIC_API_KEY"):
            return self._mock(messages)

        system = next((m["content"] for m in messages if m["role"] == "system"), None)
        turns = [m for m in messages if m["role"] != "system"]

        client = AsyncAnthropic()
        resp = await client.messages.create(
            model=model,
            system=system,
            messages=turns,
            max_tokens=1024,
        )
        content = "".join(block.text for block in resp.content if block.type == "text")
        return ProviderResult(
            content=content,
            prompt_tokens=resp.usage.input_tokens,
            completion_tokens=resp.usage.output_tokens,
        )

    def _mock(self, messages: list[dict]) -> ProviderResult:
        last_user = next((m["content"] for m in reversed(messages) if m["role"] == "user"), "")
        return ProviderResult(
            content=f"[mock anthropic response] {last_user[:80]}",
            prompt_tokens=max(1, len(last_user) // 4),
            completion_tokens=20,
        )
