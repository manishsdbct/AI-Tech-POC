# LLM Gateway — Design

Central FastAPI service every internal service calls instead of hitting
OpenAI/Anthropic directly. One request path handles auth, rate limiting,
response caching, cost-aware model routing with fallback, and usage/cost
tracking.

## 1. Architecture

```
                       ┌────────────────────────────────────────────────────────┐
                       │                     POST /v1/completions                │
                       └────────────────────────────────────────────────────────┘
                                              │
   1. auth.py          service_key → ServiceIdentity(service_id, team_id,        401 if unknown key
                        requests_per_min, tokens_per_min,
                        default_model_tier, max_model_tier)      ── Postgres: services table (§4)
                                              │
   2. rate_limiter.py   request/min token bucket, keyed by service_id,           429 if exhausted
                         capacity = identity.requests_per_min or gateway default
                                              │
   3. budgets.py         team spend vs budget  ── Postgres: teams table (§4)      402 if over budget
                          (skipped entirely if quality_critical=true)            (tier downgraded one
                                              │                                   step if over alert %)
   4. router.py          identity.default_model_tier / max_model_tier /
                          model_tier / model → concrete model (post-downgrade)
                                              │
   5. cache.py           exact-hash lookup on (model, messages)          ── hit ──► return cached response
                                              │ miss
   6. token_counter.py   estimate prompt tokens for the resolved messages
                                              │
   7. rate_limiter.py    tokens/min token bucket, keyed by service_id,           429 if exhausted
                          capacity = identity.tokens_per_min or gateway default
                                              │
   8. router.py           call provider adapter for the model; on failure,
      providers/*.py      walk the static fallback chain                        502 if all fail
                                              │
   9. pricing.py          cost_usd = f(model, real prompt/completion tokens
                                        returned by the provider)
                                              │
  10. cost_tracker.py     append usage_ledger row (Postgres, §4),
      observability.py    add to team running spend, Prometheus counters/
                           histograms + structured log
                                              │
                                    CompletionResponse
```

Each numbered stage is one module with one job; `main.py` (`create_completion`)
is just the glue that calls them in order. Swapping a stage's internals
(e.g. in-memory → Redis, static pricing table → pricing service) never
touches the others because they only talk through the dataclasses/pydantic
models defined at each boundary (`ProviderResult`, `RateLimitResult`,
`BudgetStatus`, `UsageRecord`, `CompletionResponse`).

**Files:**

| File | Responsibility |
|---|---|
| `main.py` | Request orchestration (the pipeline above), `/healthz`, `/metrics`, `/v1/usage/{team_id}` |
| `db.py` | Shared asyncpg connection pool (lazy init, applies `db/schema.sql`). Returns `None` if Postgres is unreachable — every caller below falls back to in-memory demo data in that case. |
| `auth.py` | `service_key` → `ServiceIdentity`, looked up by key hash against the `services` table (§4). Falls back to an in-memory demo registry if Postgres is down. |
| `rate_limiter.py` | `TokenBucketLimiter` — Redis-backed with an in-process fallback. Used twice per request: once for request volume, once for token volume. `check()` takes an optional per-call `capacity`/`refill_per_sec` override, used to apply a service's `requests_per_min`/`tokens_per_min` config rule. |
| `budgets.py` | Per-team budget lookup (`teams` table, §4) + the downgrade/hard-stop policy (§3). |
| `router.py` | `resolve_primary_model` (tier/model → concrete model) and `route_and_complete` (tries the primary, then its static fallback chain, on provider exceptions). |
| `pricing.py` | Static `PRICING` table (USD/1K tokens), `TIER_DEFAULTS`, `FALLBACKS`, `compute_cost_usd`, `TIER_ORDER`/`cap_tier` (for `max_model_tier`). |
| `providers/*.py` | One adapter per vendor SDK, normalized to `ProviderResult{content, prompt_tokens, completion_tokens}`. Each falls back to a mock response if the SDK isn't installed *or* its API key isn't set, so the whole path is exercisable offline. |
| `cache.py` | Exact-hash (`sha256(model + messages)`) response cache. Redis-backed with an in-process fallback. |
| `token_counter.py` | Pre-call *estimate* of prompt tokens (§2) — not the billing source of truth. |
| `cost_tracker.py` | Writes each request to the `usage_ledger` table (§4) and reads team spend back from it (`SUM(cost_usd)`). Falls back to an in-memory ledger/running total if Postgres is down. |
| `observability.py` | Prometheus metrics + structured request logs. |

**Resilience pattern used throughout:** every external dependency (Redis,
Postgres, the OpenAI/Anthropic SDKs) is optional at the code level — each
integration point tries the real thing and falls back to an in-process/mock
equivalent on `ImportError` or a connection failure. That's what makes
`uvicorn app.main:app --reload` runnable with zero infra. When Postgres
*is* configured and reachable, it's the source of truth for identity,
config rules, and spend — the in-memory registries only matter for offline
dev.

## 2. Token calculation

Two different token numbers exist in the system, for two different purposes:

**Before the call — estimate (`token_counter.py`).** The gateway needs to
know roughly how many tokens a request will cost *before* spending a
provider call, so the tokens-per-minute limiter can reject an oversized
request cheaply. `estimate_prompt_tokens(messages)`:

- Uses `tiktoken`'s `cl100k_base` encoding if it's importable and its BPE
  ranks are already cached locally (lazy import, exactly like the provider
  SDKs — see `_get_encoding()`).
- Otherwise falls back to a `len(text) // 4` heuristic (rough
  chars-per-token average for English).
- Either way, adds OpenAI's standard per-message chat overhead (4 tokens/
  message + a 3-token reply primer) so the estimate is structurally
  consistent with how chat-format tokenizers actually count, even though
  the exact vendor-specific framing differs slightly between OpenAI and
  Anthropic.

This estimate is deliberately conservative-but-cheap — it's a gate, not a
bill.

**After the call — actuals (`providers/*.py`, `pricing.py`).** Every
provider adapter returns the vendor's own reported `prompt_tokens` /
`completion_tokens` (`resp.usage.input_tokens` etc. for Anthropic,
`resp.usage.prompt_tokens` etc. for OpenAI). In mock mode (no API key
configured), the adapter approximates with the same `len(text) // 4`
heuristic for the prompt side and a fixed `20` for the completion side.
**Cost is always computed from these actual/mock post-call numbers**
(`pricing.compute_cost_usd`), never from the pre-call estimate — the
estimate only gates the token-rate limiter.

```
cost_usd = (prompt_tokens / 1000) * PRICING[model].prompt_per_1k
         + (completion_tokens / 1000) * PRICING[model].completion_per_1k
```

**Where each number is used:**

| Number | Source | Used for |
|---|---|---|
| Estimated prompt tokens | `token_counter.estimate_prompt_tokens` | Charging the tokens/min bucket before the provider call |
| Actual prompt/completion tokens | Provider response (or mock heuristic) | `cost_usd`, the `usage_ledger` row, `Usage` in the API response, `gateway_tokens_total` metric |

The token-rate bucket is charged against the *estimate*, not trued up
against the actual count afterward — acceptable slop for a rate limiter
(it only needs to prevent sustained overrun, not be exact), but worth
knowing if you're debugging why the bucket drained faster/slower than a
token count you computed by hand.

## 3. Cost optimization logic

Two independent levers, both driven by `model_tier` + the per-request
`quality_critical` flag:

### 3.1 Tier → model resolution (`router.resolve_primary_model`)

Callers pick a `model_tier` (`cheap` / `standard` / `premium`) rather than
a hardcoded model string. `pricing.TIER_DEFAULTS` maps each tier to a
concrete model, so the cost/quality tradeoff for "cheap" can be repointed
(e.g. Haiku → a newer cheaper model) without any caller changing code. An
explicit `model` field always wins over `model_tier` for callers that need
a specific model.

Before that resolution happens, `main.py` applies each service's own tier
policy (from the `services` table, §4):

- If the caller didn't specify `model_tier`, the service's
  `default_model_tier` is used instead of a hardcoded `"standard"` — e.g.
  the seeded `support-bot` service defaults to `cheap` rather than
  `standard`.
- If the service has a `max_model_tier` set, the effective tier is capped
  to it (`pricing.cap_tier`) even if the caller explicitly asked for
  something more expensive — e.g. `support-bot` is capped at `standard`
  and will never be routed to `premium`, regardless of what it requests.
- Both only apply to tier-based requests, same carve-out as the budget
  downgrade below: an explicit `model` bypasses tier policy entirely.

### 3.2 Budget-aware downgrade & hard stop (`budgets.py`, wired into `main.py`)

Each team has a monthly USD budget (`budgets._TEAM_BUDGETS`, demo data —
replace with the `budgets` table). On every request, before routing:

```
ratio = team_spend_usd / team_budget_usd
over_alert  = ratio >= settings.budget_alert_threshold_pct   # default 0.8
over_budget = team_spend_usd >= team_budget_usd
```

- **`over_budget` and not `quality_critical`** → **hard stop**: reject with
  `402 Payment Required` before any provider call or cache lookup. The
  caller can retry with `quality_critical=true` if the request genuinely
  can't wait.
- **`over_alert` (but not yet over budget) and not `quality_critical` and no
  explicit `model` was requested** → **downgrade**: the tier steps down one
  notch (`premium→standard→cheap`; `cheap` stays `cheap`) via
  `budgets.downgrade_tier`, and *that* downgraded tier is what gets resolved
  to a model, cached against, and billed. The response's
  `downgraded_from_tier` field reports the original tier so callers/UI can
  see it happened.
- **`quality_critical=true`** bypasses both checks entirely — a team being
  over budget is a policy problem for whoever owns that budget, not a
  reason to silently degrade or drop a request the caller has explicitly
  flagged as critical.
- An explicit `model` (rather than `model_tier`) is never downgraded (there's
  no defined "one step cheaper" for an arbitrary model string) but **is**
  still subject to the hard stop.

This is intentionally a one-step-down policy per request rather than an
immediate jump to the cheapest tier: it eases spend down proportionally to
how much traffic a team is sending while over its alert threshold, instead
of cliff-dropping quality for every caller the moment the threshold trips.

### 3.3 Other cost levers already in the pipeline

- **Exact-match response caching** (`cache.py`): identical `(model,
  messages)` requests within `GATEWAY_CACHE_TTL_SECONDS` skip the provider
  entirely — no tokens billed, and cache hits also skip the tokens/min
  check (§2) since no provider call happens. Semantic (embedding-similarity)
  caching is the natural v2 extension, noted in `cache.py`.
- **Fallback chain** (`pricing.FALLBACKS`, `router.route_and_complete`):
  triggers on provider *errors*, not cost — a `claude-opus-4` failure
  falls back to `claude-sonnet-5` then `gpt-4o`, which happens to also be
  cheaper, but the chain exists for availability, not budget.
- **Per-team cost visibility** (`GET /v1/usage/{team_id}`, `COST_TOTAL`
  Prometheus metric): lets a team/dashboard see spend against budget in
  real time, independent of whether the gateway is actively throttling
  them.

## 4. Data & config: Postgres

Schema lives in [`db/schema.sql`](db/schema.sql) — three tables:

- **`teams`**: `team_id`, `name`, `monthly_budget_usd`. Read by `budgets.py`.
- **`services`**: `service_id`, `service_key_hash` (sha256 — the plaintext
  key is never stored), `team_id`, plus the per-service config rules:
  `requests_per_min`, `tokens_per_min` (NULL = use the gateway-wide
  default), `default_model_tier`, `max_model_tier`. Read by `auth.py`.
- **`usage_ledger`**: one row per completed request (billing actuals, §2),
  written by `cost_tracker.py`. `team_spend_usd` is `SUM(cost_usd)` over
  this table, so budget checks always reflect real persisted spend.

`app/db.py` applies the schema automatically (idempotent `CREATE TABLE IF
NOT EXISTS`) the first time any request needs the pool.

**Seeding sample data:** `python scripts/seed_db.py` upserts three demo
teams and three demo services, each exercising a different combination of
config rules:

| service_id | service_key | team | requests/tokens per min | default → max tier |
|---|---|---|---|---|
| `search` | `svc-search-key` | `platform` ($50 budget) | gateway default | `standard`, uncapped |
| `support-bot` | `svc-support-key` | `cx` ($20 budget) | 300 / 100,000 | `cheap` → capped at `standard` |
| `recommender` | `svc-reco-key` | `growth` ($15 budget) | 120 / 50,000 | `cheap` → capped at `cheap` |

Point `GATEWAY_POSTGRES_DSN` (`.env`) at your Postgres instance before
seeding — `docker compose up` brings up a matching one, or run Postgres
locally and match `.env.example`'s default DSN
(`postgresql://postgres:postgres@localhost:5432/llm_gateway`).

If Postgres is unreachable, every module above falls back to a small
hardcoded in-memory registry (same two `service_key`s from the original
scaffold, `svc-search-key`/`svc-support-key`, plus the `platform`/`cx`
budgets) so local dev without any DB still works — see §1's resilience
note.

## 5. Next steps to make this production-ready

- True up the tokens/min bucket against actual usage after the call, not
  just the pre-call estimate, if precise enforcement matters more than
  bucket-drain performance.
- Add semantic caching (embedding similarity) alongside the exact-hash
  cache in `cache.py`.
- Add OpenTelemetry tracing spans around the provider call in `router.py`.
- Add per-team aggregate rate limiting (today's `rate_limiter` is
  per-service only).
- Budget period reset (`teams.monthly_budget_usd` is currently a flat
  ceiling with no monthly rollover — spend accumulates forever).
- Service key rotation/revocation flow (`services.is_active` exists but
  nothing sets it yet) and moving key verification to a slower hash
  (bcrypt/argon2) if keys need to resist offline brute force.
