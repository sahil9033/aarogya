"""Cache abstraction with a Redis backend and an in-process fallback.

Both implementations expose the same async surface, so callers never branch on
which one is active. The fallback is process-local: correct for a single
instance, and it degrades to a cache miss (never a wrong answer) when scaled
out without Redis configured.
"""

from __future__ import annotations

import asyncio
import json
import time
from typing import Any, Protocol, runtime_checkable

from app.core.config import settings
from app.core.logging import get_logger

logger = get_logger(__name__)


@runtime_checkable
class Cache(Protocol):
    @property
    def backend_name(self) -> str: ...

    async def get(self, key: str) -> Any | None: ...

    async def set(self, key: str, value: Any, ttl: int | None = None) -> None: ...

    async def delete(self, key: str) -> None: ...

    async def close(self) -> None: ...


class MemoryCache:
    """TTL cache backed by a dict. Suitable for development and tests."""

    backend_name = "memory"

    def __init__(self) -> None:
        self._store: dict[str, tuple[float | None, Any]] = {}
        self._lock = asyncio.Lock()

    async def get(self, key: str) -> Any | None:
        async with self._lock:
            entry = self._store.get(key)
            if entry is None:
                return None
            expires_at, value = entry
            if expires_at is not None and expires_at < time.monotonic():
                self._store.pop(key, None)
                return None
            return value

    async def set(self, key: str, value: Any, ttl: int | None = None) -> None:
        expires_at = None
        if ttl:
            expires_at = time.monotonic() + ttl
        async with self._lock:
            self._store[key] = (expires_at, value)

    async def delete(self, key: str) -> None:
        async with self._lock:
            self._store.pop(key, None)

    async def close(self) -> None:
        async with self._lock:
            self._store.clear()


class RedisCache:
    backend_name = "redis"

    def __init__(self, client) -> None:
        self._client = client

    async def get(self, key: str) -> Any | None:
        raw = await self._client.get(key)
        if raw is None:
            return None
        try:
            return json.loads(raw)
        except (TypeError, ValueError):
            return raw

    async def set(self, key: str, value: Any, ttl: int | None = None) -> None:
        await self._client.set(key, json.dumps(value, default=str), ex=ttl)

    async def delete(self, key: str) -> None:
        await self._client.delete(key)

    async def close(self) -> None:
        await self._client.aclose()


_cache: Cache | None = None


async def create_cache() -> Cache:
    """Build the best available cache and verify the connection."""
    global _cache
    if _cache is not None:
        return _cache

    if settings.redis_url:
        try:
            from redis.asyncio import from_url

            client = from_url(settings.redis_url, encoding="utf-8", decode_responses=True)
            await client.ping()
            _cache = RedisCache(client)
            logger.info("cache backend ready", extra={"backend": "redis"})
            return _cache
        except Exception as exc:
            logger.warning("redis unavailable, falling back to in-process cache: %s", exc)

    _cache = MemoryCache()
    logger.info("cache backend ready", extra={"backend": "memory"})
    return _cache


def get_cache() -> Cache:
    """Return the active cache. Falls back to memory if startup did not run."""
    global _cache
    if _cache is None:
        _cache = MemoryCache()
    return _cache


async def close_cache() -> None:
    global _cache
    if _cache is not None:
        await _cache.close()
    _cache = None


def cache_key(*parts: str) -> str:
    return "aarogya:" + ":".join(str(part) for part in parts)
