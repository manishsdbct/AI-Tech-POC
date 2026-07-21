"""Central configuration for the LLM gateway.

All tunables live here so behavior can change via env vars without code edits.
"""
from functools import lru_cache
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    redis_url: str = "redis://localhost:6379/0"
    postgres_dsn: str = "postgresql://postgres:postgres@localhost:5432/llm_gateway"

    # Rate limiting defaults (overridable per-service in the DB)
    default_requests_per_min: int = 600
    default_tokens_per_min: int = 200_000

    # Cache
    cache_enabled: bool = True
    cache_ttl_seconds: int = 3600
    semantic_cache_similarity_threshold: float = 0.95

    # Budgets
    budget_alert_threshold_pct: float = 0.8

    class Config:
        env_prefix = "GATEWAY_"
        env_file = ".env"


@lru_cache
def get_settings() -> Settings:
    return Settings()
