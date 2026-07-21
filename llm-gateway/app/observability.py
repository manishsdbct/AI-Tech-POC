"""Structured logging + Prometheus metrics for every gateway request.

Metrics are exposed at GET /metrics for scraping. Swap the logger's handler
for JSON output (e.g. via python-json-logger) in production.
"""
import logging

from prometheus_client import Counter, Histogram

logger = logging.getLogger("gateway")
logging.basicConfig(level=logging.INFO)

REQUEST_COUNT = Counter(
    "gateway_requests_total", "Total completion requests", ["service_id", "model", "status"]
)
REQUEST_LATENCY = Histogram(
    "gateway_request_latency_seconds", "Request latency", ["service_id", "model"]
)
TOKENS_TOTAL = Counter(
    "gateway_tokens_total", "Tokens processed", ["service_id", "model", "kind"]
)
COST_TOTAL = Counter(
    "gateway_cost_usd_total", "Cost accrued in USD", ["service_id", "team_id", "model"]
)
CACHE_HITS = Counter("gateway_cache_hits_total", "Cache hits", ["service_id"])
RATE_LIMIT_REJECTIONS = Counter(
    "gateway_rate_limit_rejections_total", "Requests rejected by rate limiter", ["service_id"]
)
TOKEN_LIMIT_REJECTIONS = Counter(
    "gateway_token_limit_rejections_total", "Requests rejected by the tokens-per-minute limiter", ["service_id"]
)
BUDGET_DOWNGRADES = Counter(
    "gateway_budget_downgrades_total",
    "Requests routed to a cheaper tier due to team budget pressure",
    ["team_id", "from_tier", "to_tier"],
)
BUDGET_BLOCKED = Counter(
    "gateway_budget_blocked_total", "Requests hard-blocked for exceeding team budget", ["team_id"]
)


def log_request(**fields) -> None:
    logger.info("request", extra={"gateway_fields": fields})
