"""Exact-match response cache, keyed by a hash of the normalized request.

Semantic caching (embedding similarity) is a v2 extension: hook it into
`get()` by checking embedding similarity against recent cache entries before
falling back to the exact-hash miss.
"""
import hashlib
import json
import time
from typing import Optional

from .config import get_settings

try:
    import redis.asyncio as redis
except ImportError:  # pragma: no cover
    redis = None


def _hash_request(model: str, messages: list[dict]) -> str:
    payload = json.dumps({"model": model, "messages": messages}, sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()


class ResponseCache:
    def __init__(self):
        self._redis = None
        self._local: dict[str, tuple[float, dict]] = {}

    async def _get_redis(self):
        if redis is None:
            return None
        if self._redis is None:
            settings = get_settings()
            self._redis = redis.from_url(settings.redis_url, decode_responses=True)
        return self._redis

    async def get(self, model: str, messages: list[dict]) -> Optional[dict]:
        settings = get_settings()
        if not settings.cache_enabled:
            return None
        key = _hash_request(model, messages)

        r = await self._get_redis()
        if r is not None:
            try:
                raw = await r.get(f"cache:{key}")
                return json.loads(raw) if raw else None
            except Exception:
                pass

        entry = self._local.get(key)
        if entry and (time.time() - entry[0]) < settings.cache_ttl_seconds:
            return entry[1]
        return None

    async def set(self, model: str, messages: list[dict], response: dict) -> None:
        settings = get_settings()
        key = _hash_request(model, messages)

        r = await self._get_redis()
        if r is not None:
            try:
                await r.set(f"cache:{key}", json.dumps(response), ex=settings.cache_ttl_seconds)
                return
            except Exception:
                pass

        self._local[key] = (time.time(), response)
