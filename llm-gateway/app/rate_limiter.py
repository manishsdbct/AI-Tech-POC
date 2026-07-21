"""Redis-backed token bucket rate limiter, per (service_id, model_tier).

Falls back to an in-process bucket if Redis is unreachable, so the scaffold
is runnable without infra during local dev.
"""
import time
from dataclasses import dataclass

try:
    import redis.asyncio as redis
except ImportError:  # pragma: no cover - redis is an optional dev dependency here
    redis = None

from .config import get_settings


@dataclass
class RateLimitResult:
    allowed: bool
    retry_after_seconds: float = 0.0


class TokenBucketLimiter:
    """Token bucket: `capacity` tokens, refilled at `refill_per_sec`."""

    def __init__(self, capacity: int, refill_per_sec: float):
        self.capacity = capacity
        self.refill_per_sec = refill_per_sec
        self._redis = None
        self._local_buckets: dict[str, tuple[float, float]] = {}  # key -> (tokens, last_ts)

    async def _get_redis(self):
        if redis is None:
            return None
        if self._redis is None:
            settings = get_settings()
            self._redis = redis.from_url(settings.redis_url, decode_responses=True)
        return self._redis

    async def check(
        self,
        key: str,
        cost: int = 1,
        capacity: float | None = None,
        refill_per_sec: float | None = None,
    ) -> RateLimitResult:
        """`capacity`/`refill_per_sec` let a caller override this limiter's
        defaults for one key (e.g. a per-service `requests_per_min`/
        `tokens_per_min` config rule from the `services` table). Each key
        already has independent bucket state, so overriding the ceiling
        per-call is safe — it doesn't affect other keys' buckets.
        """
        capacity = self.capacity if capacity is None else capacity
        refill_per_sec = self.refill_per_sec if refill_per_sec is None else refill_per_sec

        r = await self._get_redis()
        if r is not None:
            try:
                return await self._check_redis(r, key, cost, capacity, refill_per_sec)
            except Exception:
                pass  # fall through to local bucket if Redis is down
        return self._check_local(key, cost, capacity, refill_per_sec)

    async def _check_redis(self, r, key: str, cost: int, capacity: float, refill_per_sec: float) -> RateLimitResult:
        now = time.time()
        bucket_key = f"ratelimit:{key}"
        pipe = r.pipeline()
        pipe.hgetall(bucket_key)
        (state,) = await pipe.execute()

        tokens = float(state.get("tokens", capacity))
        last_ts = float(state.get("ts", now))
        tokens = min(capacity, tokens + (now - last_ts) * refill_per_sec)

        if tokens < cost:
            deficit = cost - tokens
            retry_after = deficit / refill_per_sec
            await r.hset(bucket_key, mapping={"tokens": tokens, "ts": now})
            return RateLimitResult(allowed=False, retry_after_seconds=retry_after)

        tokens -= cost
        await r.hset(bucket_key, mapping={"tokens": tokens, "ts": now})
        await r.expire(bucket_key, 3600)
        return RateLimitResult(allowed=True)

    def _check_local(self, key: str, cost: int, capacity: float, refill_per_sec: float) -> RateLimitResult:
        now = time.time()
        tokens, last_ts = self._local_buckets.get(key, (capacity, now))
        tokens = min(capacity, tokens + (now - last_ts) * refill_per_sec)

        if tokens < cost:
            deficit = cost - tokens
            retry_after = deficit / refill_per_sec
            self._local_buckets[key] = (tokens, now)
            return RateLimitResult(allowed=False, retry_after_seconds=retry_after)

        tokens -= cost
        self._local_buckets[key] = (tokens, now)
        return RateLimitResult(allowed=True)


def build_request_limiter() -> TokenBucketLimiter:
    settings = get_settings()
    return TokenBucketLimiter(
        capacity=settings.default_requests_per_min,
        refill_per_sec=settings.default_requests_per_min / 60,
    )


def build_token_limiter() -> TokenBucketLimiter:
    settings = get_settings()
    return TokenBucketLimiter(
        capacity=settings.default_tokens_per_min,
        refill_per_sec=settings.default_tokens_per_min / 60,
    )
