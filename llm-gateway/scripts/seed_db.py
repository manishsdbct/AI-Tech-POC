"""Seeds Postgres with sample teams + services for local dev.

Applies db/schema.sql (idempotent) then upserts a small set of demo teams
and services, each with a different combination of config rules
(rate limit overrides, default/max model tier) so the gateway's per-service
policy behavior is exercisable out of the box.

Service keys are hashed (sha256) before being stored — same as how
auth.py looks them up — so the plaintext keys only ever exist here, in
this script's output, and in whatever client uses them.

Usage:
    python scripts/seed_db.py
"""
import asyncio
import hashlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import asyncpg  # noqa: E402

from app.config import get_settings  # noqa: E402

SCHEMA_PATH = Path(__file__).resolve().parents[1] / "db" / "schema.sql"

TEAMS = [
    # team_id,   name,                 monthly_budget_usd
    ("platform", "Platform Engineering", 50.0),
    ("cx", "Customer Experience", 20.0),
    ("growth", "Growth & Recommendations", 15.0),
]

# service_id, service_key (plaintext, hashed before storage), team_id, name,
# requests_per_min, tokens_per_min, default_model_tier, max_model_tier
SERVICES = [
    (
        "search", "svc-search-key", "platform", "Search relevance service",
        None, None, "standard", None,
    ),
    (
        "support-bot", "svc-support-key", "cx", "Customer support chatbot",
        300, 100_000, "cheap", "standard",
    ),
    (
        "recommender", "svc-reco-key", "growth", "Product recommendation ranker",
        120, 50_000, "cheap", "cheap",
    ),
]


def hash_key(key: str) -> str:
    return hashlib.sha256(key.encode()).hexdigest()


async def main() -> None:
    settings = get_settings()
    conn = await asyncpg.connect(settings.postgres_dsn)
    try:
        await conn.execute(SCHEMA_PATH.read_text())

        for team_id, name, budget in TEAMS:
            await conn.execute(
                """
                INSERT INTO teams (team_id, name, monthly_budget_usd)
                VALUES ($1, $2, $3)
                ON CONFLICT (team_id) DO UPDATE
                    SET name = EXCLUDED.name, monthly_budget_usd = EXCLUDED.monthly_budget_usd
                """,
                team_id, name, budget,
            )

        for service_id, key, team_id, name, rpm, tpm, default_tier, max_tier in SERVICES:
            await conn.execute(
                """
                INSERT INTO services
                    (service_id, service_key_hash, team_id, name,
                     requests_per_min, tokens_per_min, default_model_tier, max_model_tier)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
                ON CONFLICT (service_id) DO UPDATE
                    SET service_key_hash = EXCLUDED.service_key_hash,
                        team_id = EXCLUDED.team_id,
                        name = EXCLUDED.name,
                        requests_per_min = EXCLUDED.requests_per_min,
                        tokens_per_min = EXCLUDED.tokens_per_min,
                        default_model_tier = EXCLUDED.default_model_tier,
                        max_model_tier = EXCLUDED.max_model_tier
                """,
                service_id, hash_key(key), team_id, name, rpm, tpm, default_tier, max_tier,
            )
    finally:
        await conn.close()

    print(f"Seeded {len(TEAMS)} teams and {len(SERVICES)} services into Postgres.\n")
    print(f"{'service_id':<14} {'team_id':<10} {'service_key (plaintext)':<20} config rules")
    for service_id, key, team_id, name, rpm, tpm, default_tier, max_tier in SERVICES:
        rules = f"default_tier={default_tier}"
        if max_tier:
            rules += f", max_tier={max_tier}"
        if rpm:
            rules += f", requests_per_min={rpm}"
        if tpm:
            rules += f", tokens_per_min={tpm}"
        print(f"{service_id:<14} {team_id:<10} {key:<20} {rules}")


if __name__ == "__main__":
    asyncio.run(main())
