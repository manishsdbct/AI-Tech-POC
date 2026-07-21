"""Service authentication and identity resolution.

Looks up `service_key` (hashed, never stored in plaintext) against the
`services` table in Postgres, which also carries each service's config
rules (rate limit overrides, default/max model tier — see db/schema.sql).
Falls back to an in-memory demo registry if Postgres isn't reachable, so
the scaffold still runs standalone without infra.
"""
import hashlib
from dataclasses import dataclass
from typing import Optional

from fastapi import HTTPException, status

from .db import get_pool


@dataclass
class ServiceIdentity:
    service_id: str
    team_id: str
    requests_per_min: Optional[int] = None
    tokens_per_min: Optional[int] = None
    default_model_tier: str = "standard"
    max_model_tier: Optional[str] = None


# Fallback demo registry, used only when Postgres is unreachable. Mirrors
# the seed data in scripts/seed_db.py's default set.
_FALLBACK_REGISTRY: dict[str, ServiceIdentity] = {
    "svc-search-key": ServiceIdentity(service_id="search", team_id="platform"),
    "svc-support-key": ServiceIdentity(
        service_id="support-bot", team_id="cx", max_model_tier="standard"
    ),
}


def _hash_key(service_key: str) -> str:
    return hashlib.sha256(service_key.encode()).hexdigest()


async def authenticate(service_key: str) -> ServiceIdentity:
    pool = await get_pool()
    if pool is not None:
        row = await pool.fetchrow(
            """
            SELECT service_id, team_id, requests_per_min, tokens_per_min,
                   default_model_tier, max_model_tier
            FROM services
            WHERE service_key_hash = $1 AND is_active
            """,
            _hash_key(service_key),
        )
        if row is not None:
            return ServiceIdentity(**dict(row))
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid service key")

    identity = _FALLBACK_REGISTRY.get(service_key)
    if identity is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid service key")
    return identity
