# LLM Gateway (scaffold)

Central FastAPI gateway that every internal service calls instead of hitting
OpenAI/Anthropic directly. Handles auth, rate limiting, response caching,
provider/model routing with fallback, cost tracking, and metrics.

See [DESIGN.md](DESIGN.md) for the full architecture, the request
pipeline, how token counts are calculated (pre-call estimate vs. post-call
billing), and the budget-aware cost optimization (tier downgrade / hard
stop) logic.

## Run locally

```bash
pip install -r requirements.txt
cp .env.example .env
uvicorn app.main:app --reload
```

Redis/Postgres are optional for local dev — the rate limiter and cache fall
back to in-process state if Redis isn't reachable, and auth/budgets/cost
tracking fall back to an in-memory demo registry if Postgres isn't reachable
(see DESIGN.md §4). Provider adapters fall back to mock responses if
`openai`/`anthropic` SDKs or API keys aren't configured, so the whole request
path is exercisable with zero external dependencies.

For a real deployment: `docker compose up` brings up Redis + Postgres
alongside the gateway.

## Database (Postgres)

Service identity, per-service config rules (rate limits, default/max model
tier), and team budgets are backed by Postgres — see [DESIGN.md §4](DESIGN.md#4-data--config-postgres)
for the schema. To use it locally:

```bash
# point GATEWAY_POSTGRES_DSN at your Postgres in .env, then:
python scripts/seed_db.py
```

This creates the tables (if they don't exist) and seeds 3 sample
teams/services with different config rules. It prints the plaintext demo
service keys — only their hash is stored in the DB.

## Try it

```bash
curl -X POST localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
        "service_key": "svc-search-key",
        "model_tier": "standard",
        "messages": [{"role": "user", "content": "Summarize this doc"}]
      }'
```

Demo service keys (seeded by `scripts/seed_db.py`, see `app/auth.py` for the
offline fallback): `svc-search-key` (team: platform), `svc-support-key`
(team: cx, capped at `standard` tier), `svc-reco-key` (team: growth, capped
at `cheap` tier).

## UI

A small console for manually exercising the gateway is served at `/` —
pick a demo service key, a model tier, and a prompt, and it hits
`/v1/completions` and shows the response, token/cost usage, and whether
the request was downgraded for budget reasons.

## Next steps to make this production-ready

See [DESIGN.md](DESIGN.md#5-next-steps-to-make-this-production-ready) for
the full list (Postgres-backed persistence, semantic caching, tracing,
per-team aggregate rate limits, budget period rollover).
