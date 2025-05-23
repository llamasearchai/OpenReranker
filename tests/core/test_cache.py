import asyncio
import hashlib
import json
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from open_reranker.core.cache import (
    CacheManager,
    InMemoryCache,
    RedisCache,
    cache_key,
    cached,
)
from open_reranker.core.config import Settings
from open_reranker.core.logging import setup_logging  # For logger in cached decorator

logger = setup_logging()  # Required for the @cached decorator tests


@pytest.fixture
def mock_settings_cache():
    return Settings(REDIS_URL=None, CACHE_ENABLED=True, CACHE_TTL=60)


@pytest.fixture
async def in_memory_cache_instance():
    cache = InMemoryCache()
    await cache.clear()  # Ensure clean state
    return cache


@pytest.fixture
async def redis_cache_mocked():
    mock_redis_client = AsyncMock()
    with patch("redis.asyncio.from_url", return_value=mock_redis_client):
        rc = RedisCache("redis://dummy")
        rc.redis = mock_redis_client
        return rc, mock_redis_client


@pytest.mark.asyncio
class TestInMemoryCache:
    async def test_set_get_delete(self, in_memory_cache_instance):
        cache = in_memory_cache_instance
        await cache.set("key1", "value1")
        assert await cache.get("key1") == "value1"
        await cache.delete("key1")
        assert await cache.get("key1") is None

    async def test_ttl_expiry(self, in_memory_cache_instance):
        cache = in_memory_cache_instance
        await cache.set("key_ttl", "value_ttl", ttl=0.01)
        assert await cache.get("key_ttl") == "value_ttl"
        await asyncio.sleep(0.02)
        # Manual cleanup might be needed if interval hasn't passed
        cache._cleanup_expired()
        assert await cache.get("key_ttl") is None

    async def test_clear(self, in_memory_cache_instance):
        cache = in_memory_cache_instance
        await cache.set("key_a", "val_a")
        await cache.set("key_b", "val_b")
        await cache.clear()
        assert await cache.get("key_a") is None
        assert await cache.get("key_b") is None

    async def test_no_ttl(self, in_memory_cache_instance):
        cache = in_memory_cache_instance
        await cache.set(
            "key_no_ttl", "value_no_ttl", ttl=0
        )  # Default or explicit 0 means no expiry by time
        await asyncio.sleep(0.01)  # Give some time
        cache._cleanup_expired()
        assert await cache.get("key_no_ttl") == "value_no_ttl"


@pytest.mark.asyncio
class TestRedisCache:
    async def test_get_hit_redis(self, redis_cache_mocked):
        rc, mock_redis = redis_cache_mocked
        mock_redis.get.return_value = json.dumps("value_redis")
        assert await rc.get("key_redis") == "value_redis"
        mock_redis.get.assert_called_once_with("key_redis")

    async def test_get_miss_redis(self, redis_cache_mocked):
        rc, mock_redis = redis_cache_mocked
        mock_redis.get.return_value = None
        assert await rc.get("key_missing_redis") is None
        mock_redis.get.assert_called_once_with("key_missing_redis")

    async def test_set_redis_with_ttl(self, redis_cache_mocked):
        rc, mock_redis = redis_cache_mocked
        await rc.set("key_set_redis", "value_set_redis", ttl=60)
        mock_redis.setex.assert_called_once_with(
            "key_set_redis", 60, json.dumps("value_set_redis")
        )

    async def test_set_redis_no_ttl(self, redis_cache_mocked):
        rc, mock_redis = redis_cache_mocked
        await rc.set("key_set_no_ttl", "value_set_no_ttl", ttl=0)
        mock_redis.set.assert_called_once_with(
            "key_set_no_ttl", json.dumps("value_set_no_ttl")
        )

    async def test_delete_redis(self, redis_cache_mocked):
        rc, mock_redis = redis_cache_mocked
        await rc.delete("key_del_redis")
        mock_redis.delete.assert_called_once_with("key_del_redis")

    async def test_clear_redis(self, redis_cache_mocked):
        rc, mock_redis = redis_cache_mocked
        await rc.clear()
        mock_redis.flushdb.assert_called_once()

    async def test_get_redis_json_decode_error(self, redis_cache_mocked):
        rc, mock_redis = redis_cache_mocked
        mock_redis.get.return_value = "this is not valid json"
        # Should log an error and return None
        with patch.object(rc.redis, "get", return_value="not json"):
            assert await rc.get("bad_json_key") is None


@pytest.mark.asyncio
class TestCacheManager:
    @patch("open_reranker.core.cache.settings")
    async def test_uses_in_memory_default(self, mock_cache_settings):
        mock_cache_settings.REDIS_URL = None
        manager = CacheManager()
        assert isinstance(manager.memory_cache, InMemoryCache)
        assert manager.redis_cache is None
        with patch.object(
            manager.memory_cache, "get", new_callable=AsyncMock
        ) as mock_mem_get:
            await manager.get("some_key")
            mock_mem_get.assert_called_once_with("some_key")

    @patch("open_reranker.core.cache.settings")
    @patch("open_reranker.core.cache.RedisCache")
    async def test_uses_redis_if_configured(self, MockRedisCache, mock_cache_settings):
        mock_cache_settings.REDIS_URL = "redis://localhost"
        mock_redis_instance = AsyncMock()
        MockRedisCache.return_value = mock_redis_instance
        manager = CacheManager()
        assert manager.redis_cache == mock_redis_instance
        await manager.get("redis_key")
        mock_redis_instance.get.assert_called_once_with("redis_key")

    @patch("open_reranker.core.cache.settings")
    async def test_cache_disabled(self, mock_cache_settings):
        mock_cache_settings.CACHE_ENABLED = False
        manager = CacheManager()
        with patch.object(
            manager.memory_cache, "get", new_callable=AsyncMock
        ) as mock_mem_get:
            with patch.object(
                manager, "redis_cache", new_callable=AsyncMock
            ) as mock_redis_get:
                assert await manager.get("any_key") is None
                mock_mem_get.assert_not_called()
                if (
                    manager.redis_cache
                ):  # Should not exist if CACHE_ENABLED is false and REDIS_URL is None
                    mock_redis_get.get.assert_not_called()  # Should not be called

    def test_get_cache_key(self, mock_settings_cache):
        with patch("open_reranker.core.cache.settings", mock_settings_cache):
            manager = CacheManager()
            key1 = manager._get_cache_key(
                "prefix", query="test", docs=["d1", "d2"], model="m1"
            )
            key2 = manager._get_cache_key(
                "prefix", query="test", docs=["d1", "d2"], model="m1"
            )
            key3 = manager._get_cache_key(
                "prefix", query="test2", docs=["d1", "d2"], model="m1"
            )
            assert key1 == key2
            assert key1 != key3
            assert key1.startswith("prefix:")

    @patch("open_reranker.core.cache.settings")
    async def test_rerank_cache_flow(self, mock_cache_settings, mock_settings_cache):
        mock_cache_settings.REDIS_URL = None  # Use in-memory for this test
        mock_cache_settings.CACHE_ENABLED = True
        manager = CacheManager()

        query, docs, model = "q1", ["doc1"], "model1"
        rerank_result = [(0, 0.9)]

        # Test miss
        assert await manager.get_rerank_result(query, docs, model) is None
        # Test set
        await manager.set_rerank_result(query, docs, model, rerank_result)
        # Test hit
        cached_val = await manager.get_rerank_result(query, docs, model)
        assert cached_val == rerank_result


@pytest.mark.asyncio
async def test_cached_decorator(mock_settings_cache):
    mock_settings_cache.REDIS_URL = None  # Ensure in-memory for predictability
    mock_settings_cache.CACHE_ENABLED = True

    with patch("open_reranker.core.cache.settings", mock_settings_cache):
        # Re-init manager to pick up patched settings for the decorator's global cache_manager
        with patch(
            "open_reranker.core.cache.cache_manager", CacheManager()
        ) as fresh_cache_manager:

            mock_func = AsyncMock(return_value="computed_value")
            mock_func.__name__ = "mock_func_for_cache"

            @cached(ttl=10, key_prefix="test_decorator")
            async def my_cached_function(arg1, kwarg1=None):
                return await mock_func(arg1, kwarg1=kwarg1)

            # First call - should compute and cache
            res1 = await my_cached_function("hello", kwarg1="world")
            assert res1 == "computed_value"
            mock_func.assert_called_once_with("hello", kwarg1="world")

            # Second call - should hit cache
            mock_func.reset_mock()
            res2 = await my_cached_function("hello", kwarg1="world")
            assert res2 == "computed_value"
            mock_func.assert_not_called()

            # Call with different args - should compute and cache
            mock_func.reset_mock()
            res3 = await my_cached_function("different_arg")
            assert res3 == "computed_value"
            mock_func.assert_called_once_with("different_arg", kwarg1=None)


def test_cache_key_generation():
    k1 = cache_key("a", 1, kwarg1="val1")
    k2 = cache_key("a", 1, kwarg1="val1")
    k3 = cache_key("b", 1, kwarg1="val1")
    k4 = cache_key("a", 2, kwarg1="val1")
    k5 = cache_key("a", 1, kwarg1="val2")

    assert k1 == k2
    assert k1 != k3
    assert k1 != k4
    assert k1 != k5
    assert isinstance(k1, str) and len(k1) == 32  # MD5 hexdigest length
