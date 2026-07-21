"""Per-team spend budgets and the tier-downgrade policy applied when a team
is over budget.

Budgets live in the `teams` table (db/schema.sql). Falls back to an
in-memory demo registry if Postgres isn't reachable, same pattern as
auth.py. `cost_tracker.team_spend_usd` (cost_tracker.py) is the source of
current spend; this module just decides what to do about it.
"""
from dataclasses import dataclass

from .config import get_settings
from .db import get_pool

# Fallback demo registry, used only when Postgres is unreachable. Mirrors
# the seed data in scripts/seed_db.py's default set.
_FALLBACK_TEAM_BUDGETS: dict[str, float] = {
    "platform": 50.0,
    "cx": 20.0,
}
_DEFAULT_BUDGET_USD = 10.0

# Tiers ordered from most to least expensive; downgrading walks right one
# step per request rather than jumping straight to "cheap", so spend eases
# down instead of cliff-dropping quality for every caller at once.
_TIER_DOWNGRADE: dict[str, str] = {
    "premium": "standard",
    "standard": "cheap",
    "cheap": "cheap",
}


@dataclass(frozen=True)
class BudgetStatus:
    team_id: str
    spend_usd: float
    budget_usd: float
    ratio: float
    over_alert: bool
    over_budget: bool


async def get_team_budget_usd(team_id: str) -> float:
    pool = await get_pool()
    if pool is not None:
        value = await pool.fetchval("SELECT monthly_budget_usd FROM teams WHERE team_id = $1", team_id)
        if value is not None:
            return float(value)
        return _DEFAULT_BUDGET_USD

    return _FALLBACK_TEAM_BUDGETS.get(team_id, _DEFAULT_BUDGET_USD)


async def check_budget(team_id: str, spend_usd: float) -> BudgetStatus:
    settings = get_settings()
    budget_usd = await get_team_budget_usd(team_id)
    ratio = spend_usd / budget_usd if budget_usd > 0 else float("inf")
    return BudgetStatus(
        team_id=team_id,
        spend_usd=spend_usd,
        budget_usd=budget_usd,
        ratio=ratio,
        over_alert=ratio >= settings.budget_alert_threshold_pct,
        over_budget=spend_usd >= budget_usd,
    )


def downgrade_tier(tier: str) -> str:
    return _TIER_DOWNGRADE.get(tier, tier)
