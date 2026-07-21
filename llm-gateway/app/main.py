"""LLM Gateway — FastAPI entrypoint.

Wires together auth -> rate limiting -> cache -> routing -> cost tracking ->
observability for every /v1/completions call. See the design doc for the
full architecture and rollout plan.
"""
import time
import uuid

from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import Response
from fastapi.staticfiles import StaticFiles
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from .auth import authenticate
from .budgets import check_budget, downgrade_tier
from .cache import ResponseCache
from .config import get_settings
from .cost_tracker import CostTracker, UsageRecord
from .models import CompletionRequest, CompletionResponse, ModelTier, Usage
from .observability import (
    BUDGET_BLOCKED,
    BUDGET_DOWNGRADES,
    CACHE_HITS,
    RATE_LIMIT_REJECTIONS,
    REQUEST_COUNT,
    REQUEST_LATENCY,
    TOKEN_LIMIT_REJECTIONS,
    TOKENS_TOTAL,
    COST_TOTAL,
    log_request,
)
from .pricing import PRICING, cap_tier, compute_cost_usd
from .rate_limiter import build_request_limiter, build_token_limiter
from .router import AllProvidersFailedError, resolve_primary_model, route_and_complete
from .token_counter import estimate_prompt_tokens

app = FastAPI(title="LLM Gateway", version="1.0.0")

request_limiter = build_request_limiter()
token_limiter = build_token_limiter()
cache = ResponseCache()
cost_tracker = CostTracker()


@app.get("/healthz")
async def healthz():
    return {"status": "ok"}


@app.get("/metrics")
async def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.post("/v1/completions", response_model=CompletionResponse)
async def create_completion(req: CompletionRequest, request: Request):
    request_id = str(uuid.uuid4())
    start = time.perf_counter()

    identity = await authenticate(req.service_key)
    settings = get_settings()

    # Per-service config rules (services.requests_per_min/tokens_per_min in
    # db/schema.sql) override the gateway-wide default when set.
    limit_key = identity.service_id
    request_capacity = identity.requests_per_min or settings.default_requests_per_min

    rl = await request_limiter.check(limit_key, capacity=request_capacity, refill_per_sec=request_capacity / 60)
    if not rl.allowed:
        RATE_LIMIT_REJECTIONS.labels(service_id=identity.service_id).inc()
        raise HTTPException(
            status_code=429,
            detail="Rate limit exceeded",
            headers={"Retry-After": str(round(rl.retry_after_seconds, 2))},
        )

    # --- Cost optimization: budget-aware hard stop / tier downgrade ---
    # `quality_critical` (see models.py) always bypasses both — critical
    # traffic shouldn't silently degrade or break because a team is over
    # budget; that's a policy decision for whoever set the budget, not this
    # request.
    budget_status = await check_budget(identity.team_id, await cost_tracker.team_spend_usd(identity.team_id))

    if budget_status.over_budget and not req.quality_critical:
        BUDGET_BLOCKED.labels(team_id=identity.team_id).inc()
        raise HTTPException(
            status_code=402,
            detail=(
                f"Team '{identity.team_id}' has exceeded its budget "
                f"(${budget_status.spend_usd:.2f} / ${budget_status.budget_usd:.2f}). "
                "Retry with quality_critical=true if this request must go through anyway."
            ),
        )

    # --- Per-service tier policy + budget downgrade ---
    # Explicit `model` requests bypass tier policy entirely (there's no
    # defined "cap"/"downgrade" for an arbitrary model string) but remain
    # subject to the hard stop above.
    effective_req = req
    downgraded_from_tier: str | None = None
    if req.model is None:
        tier = req.model_tier.value if req.model_tier else identity.default_model_tier
        if identity.max_model_tier:
            tier = cap_tier(tier, identity.max_model_tier)
        if budget_status.over_alert and not req.quality_critical:
            downgraded_tier = downgrade_tier(tier)
            if downgraded_tier != tier:
                BUDGET_DOWNGRADES.labels(
                    team_id=identity.team_id, from_tier=tier, to_tier=downgraded_tier
                ).inc()
                downgraded_from_tier = tier
                tier = downgraded_tier
        effective_req = req.model_copy(update={"model_tier": ModelTier(tier)})

    messages = [m.model_dump() for m in effective_req.messages]
    primary_model = resolve_primary_model(effective_req)

    cached = await cache.get(primary_model, messages)
    if cached is not None:
        CACHE_HITS.labels(service_id=identity.service_id).inc()
        return CompletionResponse(**cached)

    # --- Token-per-minute limit: estimate prompt tokens before spending a
    # provider call on a request that would blow the budget anyway ---
    estimated_tokens = estimate_prompt_tokens(messages)
    token_capacity = identity.tokens_per_min or settings.default_tokens_per_min
    tl = await token_limiter.check(
        limit_key, cost=estimated_tokens, capacity=token_capacity, refill_per_sec=token_capacity / 60
    )
    if not tl.allowed:
        TOKEN_LIMIT_REJECTIONS.labels(service_id=identity.service_id).inc()
        raise HTTPException(
            status_code=429,
            detail=f"Token rate limit exceeded (~{estimated_tokens} estimated prompt tokens)",
            headers={"Retry-After": str(round(tl.retry_after_seconds, 2))},
        )

    try:
        model_used, result = await route_and_complete(effective_req)
    except AllProvidersFailedError as exc:
        raise HTTPException(status_code=502, detail=f"All providers/fallbacks failed: {exc}") from exc

    provider = PRICING[model_used].provider
    cost_usd = compute_cost_usd(model_used, result.prompt_tokens, result.completion_tokens)

    await cost_tracker.record(
        UsageRecord(
            request_id=request_id,
            service_id=identity.service_id,
            team_id=identity.team_id,
            model=model_used,
            provider=provider,
            prompt_tokens=result.prompt_tokens,
            completion_tokens=result.completion_tokens,
            cost_usd=cost_usd,
            cache_hit=False,
        )
    )

    response = CompletionResponse(
        id=request_id,
        model_used=model_used,
        provider=provider,
        content=result.content,
        usage=Usage(
            prompt_tokens=result.prompt_tokens,
            completion_tokens=result.completion_tokens,
            total_tokens=result.prompt_tokens + result.completion_tokens,
            cost_usd=cost_usd,
        ),
        cache_hit=False,
        downgraded_from_tier=downgraded_from_tier,
    )

    await cache.set(primary_model, messages, response.model_dump())

    elapsed = time.perf_counter() - start
    REQUEST_COUNT.labels(service_id=identity.service_id, model=model_used, status="ok").inc()
    REQUEST_LATENCY.labels(service_id=identity.service_id, model=model_used).observe(elapsed)
    TOKENS_TOTAL.labels(service_id=identity.service_id, model=model_used, kind="prompt").inc(result.prompt_tokens)
    TOKENS_TOTAL.labels(service_id=identity.service_id, model=model_used, kind="completion").inc(result.completion_tokens)
    COST_TOTAL.labels(service_id=identity.service_id, team_id=identity.team_id, model=model_used).inc(cost_usd)

    log_request(
        request_id=request_id,
        service_id=identity.service_id,
        team_id=identity.team_id,
        model=model_used,
        latency_s=elapsed,
        cost_usd=cost_usd,
    )

    return response


@app.get("/v1/usage/{team_id}")
async def get_team_usage(team_id: str):
    settings = get_settings()
    spend = await cost_tracker.team_spend_usd(team_id)
    return {
        "team_id": team_id,
        "spend_usd": round(spend, 4),
        "alert_threshold_pct": settings.budget_alert_threshold_pct,
    }


# Serves the console UI (index.html) at "/". Mounted last so it doesn't
# shadow the API routes registered above.
app.mount("/", StaticFiles(directory=Path(__file__).parent / "static", html=True), name="static")
