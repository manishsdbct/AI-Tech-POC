"""Usage/cost ledger. Writes one row per completed request to the
`usage_ledger` table in Postgres (db/schema.sql). Falls back to an
in-memory list/running total if Postgres isn't reachable, same pattern as
auth.py/budgets.py, so team spend is still trackable (for the life of this
process) without infra.
"""
import logging
import time
import uuid
from dataclasses import dataclass, field

from .db import get_pool

logger = logging.getLogger("gateway.cost_tracker")


@dataclass
class UsageRecord:
    request_id: str
    service_id: str
    team_id: str
    model: str
    provider: str
    prompt_tokens: int
    completion_tokens: int
    cost_usd: float
    cache_hit: bool
    created_at: float = field(default_factory=time.time)


class CostTracker:
    def __init__(self):
        self._ledger: list[UsageRecord] = []
        self._team_spend: dict[str, float] = {}

    async def record(self, entry: UsageRecord) -> None:
        # Always kept as the fallback source of truth, and for `recent()`.
        self._ledger.append(entry)
        self._team_spend[entry.team_id] = self._team_spend.get(entry.team_id, 0.0) + entry.cost_usd

        pool = await get_pool()
        if pool is None:
            return
        try:
            await pool.execute(
                """
                INSERT INTO usage_ledger
                    (request_id, service_id, team_id, model, provider,
                     prompt_tokens, completion_tokens, cost_usd, cache_hit)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
                ON CONFLICT (request_id) DO NOTHING
                """,
                uuid.UUID(entry.request_id),
                entry.service_id,
                entry.team_id,
                entry.model,
                entry.provider,
                entry.prompt_tokens,
                entry.completion_tokens,
                entry.cost_usd,
                entry.cache_hit,
            )
        except Exception as exc:
            logger.warning("failed to persist usage_ledger row, kept in-memory only: %s", exc)

    async def team_spend_usd(self, team_id: str) -> float:
        pool = await get_pool()
        if pool is not None:
            try:
                value = await pool.fetchval(
                    "SELECT COALESCE(SUM(cost_usd), 0) FROM usage_ledger WHERE team_id = $1", team_id
                )
                return float(value)
            except Exception as exc:
                logger.warning("failed to read team spend from Postgres, using in-memory total: %s", exc)

        return self._team_spend.get(team_id, 0.0)

    def recent(self, limit: int = 50) -> list[UsageRecord]:
        return self._ledger[-limit:]
