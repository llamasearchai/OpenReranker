import hashlib
import json
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

import redis.asyncio as redis

from open_reranker.core.config import settings
from open_reranker.core.logging import setup_logging

logger = setup_logging()


class InMemoryCache:
    """In-memory cache with TTL support."""

    def __init__(self):
        self.cache: Dict[str, Tuple[Any, float]] = {}
        self.last_cleanup = time.time()
        self.cleanup_interval = 300  # 5 minutes

    def _cleanup_expired(self):
        """Remove expired entries."""
        now = time.time()
        if now - self.last_cleanup < self.cleanup_interval:
            return

        expired_keys = [
            key
            for key, (_, expiry) in self.cache.items()
            if expiry > 0 and now > expiry
        ]

        for key in expired_keys:
            del self.cache[key]

        self.last_cleanup = now

    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        self._cleanup_expired()

        if key not in self.cache:
            return None

        value, expiry = self.cache[key]

        # Check if expired
        if expiry > 0 and time.time() > expiry:
            del self.cache[key]
            return None

        return value

    async def set(self, key: str, value: Any, ttl: int = 0) -> None:
        """Set value in cache with optional TTL."""
        expiry = time.time() + ttl if ttl > 0 else 0
        self.cache[key] = (value, expiry)

    async def delete(self, key: str) -> None:
        """Delete key from cache."""
        self.cache.pop(key, None)

    async def clear(self) -> None:
        """Clear all cache entries."""
        self.cache.clear()


class RedisCache:
    """Redis-based cache."""

    def __init__(self, redis_url: str):
        self.redis = redis.from_url(redis_url)

    async def get(self, key: str) -> Optional[Any]:
        """Get value from Redis cache."""
        try:
            value = await self.redis.get(key)
            if value is None:
                return None
            return json.loads(value)
        except Exception as e:
            logger.error(f"Redis get error: {e}")
            return None

    async def set(self, key: str, value: Any, ttl: int = 0) -> None:
        """Set value in Redis cache with optional TTL."""
        try:
            serialized = json.dumps(value, default=str)
            if ttl > 0:
                await self.redis.setex(key, ttl, serialized)
            else:
                await self.redis.set(key, serialized)
        except Exception as e:
            logger.error(f"Redis set error: {e}")

    async def delete(self, key: str) -> None:
        """Delete key from Redis cache."""
        try:
            await self.redis.delete(key)
        except Exception as e:
            logger.error(f"Redis delete error: {e}")

    async def clear(self) -> None:
        """Clear all cache entries (use with caution)."""
        try:
            await self.redis.flushdb()
        except Exception as e:
            logger.error(f"Redis clear error: {e}")


class CacheManager:
    """Main cache manager that uses Redis if available, otherwise in-memory."""

    def __init__(self):
        self.redis_cache = None
        self.memory_cache = InMemoryCache()

        if settings.REDIS_URL and settings.CACHE_ENABLED:
            try:
                self.redis_cache = RedisCache(settings.REDIS_URL)
                logger.info("Using Redis for caching")
            except Exception as e:
                logger.warning(
                    f"Failed to connect to Redis, using in-memory caching: {e}"
                )
        else:
            logger.info("Using in-memory caching")

    def _get_cache_key(self, prefix: str, **kwargs) -> str:
        """Generate cache key from parameters."""
        # Create a deterministic hash of the parameters
        key_data = json.dumps(kwargs, sort_keys=True, default=str)
        key_hash = hashlib.md5(key_data.encode()).hexdigest()
        return f"{prefix}:{key_hash}"

    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        if not settings.CACHE_ENABLED:
            return None

        cache = self.redis_cache if self.redis_cache else self.memory_cache
        return await cache.get(key)

    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """Set value in cache."""
        if not settings.CACHE_ENABLED:
            return

        if ttl is None:
            ttl = settings.CACHE_TTL

        cache = self.redis_cache if self.redis_cache else self.memory_cache
        await cache.set(key, value, ttl)

    async def delete(self, key: str) -> None:
        """Delete key from cache."""
        cache = self.redis_cache if self.redis_cache else self.memory_cache
        await cache.delete(key)

    async def clear(self) -> None:
        """Clear all cache entries."""
        cache = self.redis_cache if self.redis_cache else self.memory_cache
        await cache.clear()

    async def get_rerank_result(
        self,
        query: str,
        documents: List[str],
        model: str,
        top_k: Optional[int] = None,
        **kwargs,
    ) -> Optional[List[Tuple[int, float]]]:
        """Get cached rerank result."""
        cache_key = self._get_cache_key(
            "rerank",
            query=query,
            documents=documents,
            model=model,
            top_k=top_k,
            **kwargs,
        )
        return await self.get(cache_key)

    async def set_rerank_result(
        self,
        query: str,
        documents: List[str],
        model: str,
        result: List[Tuple[int, float]],
        top_k: Optional[int] = None,
        ttl: Optional[int] = None,
        **kwargs,
    ) -> None:
        """Cache rerank result."""
        cache_key = self._get_cache_key(
            "rerank",
            query=query,
            documents=documents,
            model=model,
            top_k=top_k,
            **kwargs,
        )
        await self.set(cache_key, result, ttl)

    async def get_model_scores(
        self, model: str, query: str, documents: List[str]
    ) -> Optional[List[float]]:
        """Get cached model scores."""
        cache_key = self._get_cache_key(
            "scores", model=model, query=query, documents=documents
        )
        return await self.get(cache_key)

    async def set_model_scores(
        self,
        model: str,
        query: str,
        documents: List[str],
        scores: List[float],
        ttl: Optional[int] = None,
    ) -> None:
        """Cache model scores."""
        cache_key = self._get_cache_key(
            "scores", model=model, query=query, documents=documents
        )
        await self.set(cache_key, scores, ttl)


# Global cache manager instance
cache_manager = CacheManager()


def cache_key(*args, **kwargs) -> str:
    """Generate a cache key from arguments."""
    key_data = json.dumps({"args": args, "kwargs": kwargs}, sort_keys=True, default=str)
    return hashlib.md5(key_data.encode()).hexdigest()


def cached(ttl: int = None, key_prefix: str = ""):
    """Decorator for caching function results."""

    def decorator(func):
        async def wrapper(*args, **kwargs):
            if not settings.CACHE_ENABLED:
                return await func(*args, **kwargs)

            # Generate cache key
            cache_key_str = f"{key_prefix}:{cache_key(*args, **kwargs)}"

            # Try to get from cache
            cached_result = await cache_manager.get(cache_key_str)
            if cached_result is not None:
                logger.debug(f"Cache hit for {func.__name__}")
                return cached_result

            # Execute function and cache result
            result = await func(*args, **kwargs)
            await cache_manager.set(cache_key_str, result, ttl)
            logger.debug(f"Cache miss for {func.__name__}, result cached")

            return result

        return wrapper

    return decorator
