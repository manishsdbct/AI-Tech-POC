"""OpenAI adapter. Requires OPENAI_API_KEY; import of the SDK is lazy so the
scaffold still runs (in mock mode) without it installed/configured.
"""
import os

from .base import ProviderAdapter, ProviderResult


class OpenAIAdapter(ProviderAdapter):
    name = "openai"

    async def complete(self, model: str, messages: list[dict]) -> ProviderResult:
        try:
            from openai import AsyncOpenAI
        except ImportError:
            return self._mock(messages)

        if not os.environ.get("OPENAI_API_KEY"):
            return self._mock(messages)

        client = AsyncOpenAI()
        resp = await client.chat.completions.create(model=model, messages=messages)
        choice = resp.choices[0].message.content or ""
        return ProviderResult(
            content=choice,
            prompt_tokens=resp.usage.prompt_tokens,
            completion_tokens=resp.usage.completion_tokens,
        )

    def _mock(self, messages: list[dict]) -> ProviderResult:
        last_user = next((m["content"] for m in reversed(messages) if m["role"] == "user"), "")
        return ProviderResult(
            content=f"[mock openai response] {last_user[:80]}",
            prompt_tokens=max(1, len(last_user) // 4),
            completion_tokens=20,
        )
