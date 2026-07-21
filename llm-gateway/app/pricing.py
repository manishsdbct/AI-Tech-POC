"""Provider/model pricing table, in USD per 1K tokens.

Kept as plain data so it can be updated (or loaded from a DB/config service)
without touching gateway logic. Figures are illustrative placeholders —
replace with your actual negotiated/list pricing.
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class ModelPricing:
    provider: str
    prompt_per_1k: float
    completion_per_1k: float


PRICING: dict[str, ModelPricing] = {
    "gpt-4o": ModelPricing("openai", 0.0025, 0.010),
    "gpt-4o-mini": ModelPricing("openai", 0.00015, 0.0006),
    "claude-opus-4": ModelPricing("anthropic", 0.015, 0.075),
    "claude-sonnet-5": ModelPricing("anthropic", 0.003, 0.015),
    "claude-haiku-4.5": ModelPricing("anthropic", 0.0008, 0.004),
}

# Which model to use for each tier when the caller only specifies a tier.
TIER_DEFAULTS: dict[str, str] = {
    "cheap": "claude-haiku-4.5",
    "standard": "claude-sonnet-5",
    "premium": "claude-opus-4",
}

# Fallback chain if the primary model/provider errors or is rate-limited upstream.
FALLBACKS: dict[str, list[str]] = {
    "claude-opus-4": ["claude-sonnet-5", "gpt-4o"],
    "claude-sonnet-5": ["gpt-4o-mini", "claude-haiku-4.5"],
    "claude-haiku-4.5": ["gpt-4o-mini"],
    "gpt-4o": ["claude-sonnet-5"],
    "gpt-4o-mini": ["claude-haiku-4.5"],
}

# Most to least expensive; used to enforce a service's `max_model_tier`
# config rule (services.max_model_tier in db/schema.sql).
TIER_ORDER: list[str] = ["premium", "standard", "cheap"]


def cap_tier(tier: str, max_tier: str) -> str:
    """Returns whichever of `tier`/`max_tier` is cheaper (later in
    TIER_ORDER). Used to enforce a per-service tier ceiling regardless of
    what the caller requested.
    """
    if tier not in TIER_ORDER or max_tier not in TIER_ORDER:
        return tier
    return tier if TIER_ORDER.index(tier) >= TIER_ORDER.index(max_tier) else max_tier


def compute_cost_usd(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    pricing = PRICING.get(model)
    if pricing is None:
        return 0.0
    return (prompt_tokens / 1000) * pricing.prompt_per_1k + (completion_tokens / 1000) * pricing.completion_per_1k
