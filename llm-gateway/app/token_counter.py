"""Pre-call token estimation, used to enforce the tokens-per-minute limit
before a provider is ever called.

Providers return authoritative prompt/completion token counts *after* the
call completes (see providers/*.py) — those are what cost is actually
billed on. This module only estimates the *prompt* side, cheaply, so the
gateway can reject an oversized request without spending provider quota.

Uses `tiktoken` (cl100k_base) when it's importable and its BPE ranks are
already cached locally; otherwise falls back to a chars-per-token
heuristic. Either path is an estimate, never the billing source of truth.
"""
from functools import lru_cache

# OpenAI's chat format costs a few tokens per message beyond its raw content
# (role/name framing) plus a fixed primer for the model's reply turn. Close
# enough as a cross-provider approximation for pre-call estimation.
_TOKENS_PER_MESSAGE = 4
_TOKENS_PER_REPLY_PRIMER = 3

_CHARS_PER_TOKEN_HEURISTIC = 4  # rough average for English text


@lru_cache(maxsize=1)
def _get_encoding():
    try:
        import tiktoken
    except ImportError:
        return None
    try:
        return tiktoken.get_encoding("cl100k_base")
    except Exception:
        # e.g. no network access to fetch the BPE rank file on first use
        return None


def _count_text_tokens(text: str) -> int:
    encoding = _get_encoding()
    if encoding is not None:
        return len(encoding.encode(text))
    return max(1, len(text) // _CHARS_PER_TOKEN_HEURISTIC)


def estimate_prompt_tokens(messages: list[dict]) -> int:
    """Estimate the prompt token count for a chat message list, before
    calling the provider. Used to charge the tokens-per-minute bucket
    up front (see rate_limiter.py) so a single oversized request can't
    blow through the limit before it's ever measured.
    """
    total = _TOKENS_PER_REPLY_PRIMER
    for message in messages:
        total += _TOKENS_PER_MESSAGE
        total += _count_text_tokens(message.get("content", ""))
    return total
