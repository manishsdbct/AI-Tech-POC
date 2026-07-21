-- LLM Gateway schema.
--
-- teams        one row per team; owns a monthly USD budget.
-- services     one row per calling service; owns its auth key (hashed) and
--              its per-service config rules (rate limits, tier policy).
-- usage_ledger one row per completed request; the durable version of
--              cost_tracker.py's in-memory ledger.
--
-- Applied automatically by app/db.py on pool startup (CREATE TABLE IF NOT
-- EXISTS, safe to run repeatedly) and by scripts/seed_db.py directly.

CREATE TABLE IF NOT EXISTS teams (
    team_id             TEXT PRIMARY KEY,
    name                TEXT NOT NULL,
    monthly_budget_usd  NUMERIC(12, 4) NOT NULL DEFAULT 10.0,
    created_at          TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- service_key is never stored in plaintext -- only its sha256 hex digest,
-- looked up by auth.py. The plaintext key is generated once at seed time
-- and shown to the operator; losing it means rotating (re-seeding) it.
CREATE TABLE IF NOT EXISTS services (
    service_id          TEXT PRIMARY KEY,
    service_key_hash     TEXT UNIQUE NOT NULL,
    team_id              TEXT NOT NULL REFERENCES teams(team_id),
    name                 TEXT NOT NULL,
    -- Per-service config rules. NULL means "use the gateway-wide default"
    -- (GATEWAY_DEFAULT_REQUESTS_PER_MIN / GATEWAY_DEFAULT_TOKENS_PER_MIN).
    requests_per_min      INTEGER,
    tokens_per_min         INTEGER,
    default_model_tier      TEXT NOT NULL DEFAULT 'standard'
                             CHECK (default_model_tier IN ('cheap', 'standard', 'premium')),
    -- Ceiling this service can never request above, regardless of what the
    -- caller asks for (e.g. a low-trust service capped at 'standard').
    -- NULL means no ceiling.
    max_model_tier          TEXT
                             CHECK (max_model_tier IN ('cheap', 'standard', 'premium')),
    is_active                BOOLEAN NOT NULL DEFAULT TRUE,
    created_at               TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS usage_ledger (
    request_id          UUID PRIMARY KEY,
    service_id           TEXT NOT NULL REFERENCES services(service_id),
    team_id               TEXT NOT NULL REFERENCES teams(team_id),
    model                 TEXT NOT NULL,
    provider              TEXT NOT NULL,
    prompt_tokens          INTEGER NOT NULL,
    completion_tokens       INTEGER NOT NULL,
    cost_usd                NUMERIC(12, 6) NOT NULL,
    cache_hit                BOOLEAN NOT NULL DEFAULT FALSE,
    created_at                TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_usage_ledger_team_created ON usage_ledger (team_id, created_at);
CREATE INDEX IF NOT EXISTS idx_usage_ledger_service_created ON usage_ledger (service_id, created_at);
