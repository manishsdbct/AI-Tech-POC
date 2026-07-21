"""Model/provider routing: resolves a caller's model/tier request into a
concrete (provider, model) call, and walks the fallback chain on failure.
"""
import logging

from .models import CompletionRequest
from .pricing import FALLBACKS, PRICING, TIER_DEFAULTS
from .providers import ADAPTERS, ProviderResult

logger = logging.getLogger("gateway.router")


class AllProvidersFailedError(Exception):
    pass


def resolve_primary_model(req: CompletionRequest) -> str:
    if req.model:
        return req.model
    tier = req.model_tier.value if req.model_tier else "standard"
    return TIER_DEFAULTS[tier]


async def route_and_complete(req: CompletionRequest) -> tuple[str, ProviderResult]:
    """Try the primary model, then fall back down the chain. Returns
    (model_actually_used, result). Raises AllProvidersFailedError if every
    option in the chain fails.
    """
    primary = resolve_primary_model(req)
    candidates = [primary] + FALLBACKS.get(primary, [])
    messages = [m.model_dump() for m in req.messages]

    last_error: Exception | None = None
    for model in candidates:
        pricing = PRICING.get(model)
        if pricing is None:
            continue
        adapter = ADAPTERS.get(pricing.provider)
        if adapter is None:
            continue
        try:
            result = await adapter.complete(model, messages)
            return model, result
        except Exception as exc:  # noqa: BLE001 - deliberately broad: try next candidate
            logger.warning("provider %s model %s failed: %s", pricing.provider, model, exc)
            last_error = exc
            continue

    raise AllProvidersFailedError(str(last_error) if last_error else "no candidates configured")
