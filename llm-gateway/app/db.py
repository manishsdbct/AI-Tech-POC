"""Postgres connection pool, shared by auth.py, budgets.py, and
cost_tracker.py.

Lazy pool creation + a module-level "is Postgres reachable" flag, matching
the resilience pattern already used for Redis in cache.py/rate_limiter.py:
callers try the DB and fall back to in-memory demo data on any connection
error, so the gateway still runs with zero infra for local dev.
"""
import logging
from pathlib import Path

try:
    import asyncpg
except ImportError:  # pragma: no cover - asyncpg is an optional dev dependency here
    asyncpg = None

from .config import get_settings

logger = logging.getLogger("gateway.db")

_SCHEMA_PATH = Path(__file__).parent.parent / "db" / "schema.sql"

_pool = None
_pool_init_failed = False


async def get_pool():
    """Returns a live asyncpg pool, or None if Postgres/asyncpg isn't
    available. Caches the failure so we don't retry a connection on every
    request when Postgres is known to be down for this process lifetime.
    """
    global _pool, _pool_init_failed

    if asyncpg is None or _pool_init_failed:
        return None
    if _pool is not None:
        return _pool

    settings = get_settings()
    try:
        _pool = await asyncpg.create_pool(settings.postgres_dsn, min_size=1, max_size=5)
        async with _pool.acquire() as conn:
            await conn.execute(_SCHEMA_PATH.read_text())
    except Exception as exc:
        logger.warning("Postgres unavailable, falling back to in-memory demo data: %s", exc)
        _pool = None
        _pool_init_failed = True
        return None

    return _pool
