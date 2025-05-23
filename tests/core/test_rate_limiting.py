import asyncio
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import HTTPException

from open_reranker.core.auth import get_current_user_optional  # For dependency testing
from open_reranker.core.config import Settings
from open_reranker.core.rate_limiting import (
    InMemoryRateLimiter,
    RateLimiter,
    RedisRateLimiter,
    check_rate_limits,
    get_client_id,
    rate_limit_dependency,
)


@pytest.fixture
def mock_settings_rate_limit():
    return Settings(
        REDIS_URL=None,  # Default to in-memory for most tests
        RATE_LIMIT_QPS=2,
        RATE_LIMIT_RPM=5,
        RATE_LIMIT_TPM=100,
        COUNT_TOKENS=True,
    )


@pytest.fixture
def mock_request():
    req = MagicMock()
    req.client.host = "127.0.0.1"
    req.headers = {}
    return req


@pytest.fixture
def mock_user():
    return {"sub": "test_user", "tier": "free"}


@pytest.fixture
async def in_memory_limiter():
    return InMemoryRateLimiter()


@pytest.fixture
async def redis_limiter_mocked():
    mock_redis_client = AsyncMock()
    mock_redis_client.pipeline.return_value = AsyncMock()  # Mock the pipeline object
    # Further mock pipeline methods like zremrangebyscore, zcard, etc. as needed

    with patch("redis.asyncio.from_url", return_value=mock_redis_client):
        limiter = RedisRateLimiter("redis://dummy")
        limiter.redis = mock_redis_client  # Ensure the instance uses the mock
        return limiter, mock_redis_client


@pytest.mark.asyncio
class TestInMemoryRateLimiter:
    async def test_is_allowed_within_limit(self, in_memory_limiter):
        key = "test_key_qps"
        assert await in_memory_limiter.is_allowed(key, limit=2, window=1) is True
        assert await in_memory_limiter.is_allowed(key, limit=2, window=1) is True
        assert await in_memory_limiter.is_allowed(key, limit=2, window=1) is False

    async def test_is_allowed_window_expiry(self, in_memory_limiter):
        key = "test_key_window"
        assert await in_memory_limiter.is_allowed(key, limit=1, window=0.01) is True
        assert await in_memory_limiter.is_allowed(key, limit=1, window=0.01) is False
        await asyncio.sleep(0.02)
        assert await in_memory_limiter.is_allowed(key, limit=1, window=0.01) is True

    async def test_is_token_allowed_within_limit(self, in_memory_limiter):
        key = "test_key_tpm"
        assert (
            await in_memory_limiter.is_token_allowed(
                key, tokens=50, limit=100, window=1
            )
            is True
        )
        assert (
            await in_memory_limiter.is_token_allowed(
                key, tokens=50, limit=100, window=1
            )
            is True
        )
        assert (
            await in_memory_limiter.is_token_allowed(key, tokens=1, limit=100, window=1)
            is False
        )

    async def test_is_token_allowed_window_expiry(self, in_memory_limiter):
        key = "test_key_token_window"
        assert (
            await in_memory_limiter.is_token_allowed(
                key, tokens=100, limit=100, window=0.01
            )
            is True
        )
        assert (
            await in_memory_limiter.is_token_allowed(
                key, tokens=1, limit=100, window=0.01
            )
            is False
        )
        await asyncio.sleep(0.02)
        assert (
            await in_memory_limiter.is_token_allowed(
                key, tokens=1, limit=100, window=0.01
            )
            is True
        )


@pytest.mark.asyncio
class TestRedisRateLimiter:
    async def test_is_allowed_redis(self, redis_limiter_mocked):
        limiter, mock_redis = redis_limiter_mocked
        key = "test_redis_qps"

        # Simulate Redis pipeline responses
        # zremrangebyscore, zcard (returns 0 initially), zadd, expire
        mock_redis.pipeline.return_value.execute.side_effect = [
            [None, 0, None, None],  # First call, zcard returns 0
            [None, 1, None, None],  # Second call, zcard returns 1
            [None, 2, None, None],  # Third call, zcard returns 2 (limit exceeded)
        ]

        assert await limiter.is_allowed(key, limit=2, window=1) is True
        assert await limiter.is_allowed(key, limit=2, window=1) is True
        assert await limiter.is_allowed(key, limit=2, window=1) is False
        assert mock_redis.pipeline.return_value.zremrangebyscore.call_count == 3
        assert mock_redis.pipeline.return_value.zcard.call_count == 3
        assert mock_redis.pipeline.return_value.zadd.call_count == 3

    async def test_is_token_allowed_redis(self, redis_limiter_mocked):
        limiter, mock_redis = redis_limiter_mocked
        key = "test_redis_tpm"

        # Simulate Redis pipeline responses for token limiting
        # zremrangebyscore, zrange (returns empty list initially), zadd, expire
        mock_redis.pipeline.return_value.execute.side_effect = [
            [None, []],  # First call, zrange returns empty list (0 tokens)
            [None, [(b"50", time.time())]],  # Second call, 50 tokens used
            [
                None,
                [(b"50", time.time()), (b"50", time.time())],
            ],  # Third call, 100 tokens used (limit met)
        ]

        assert (
            await limiter.is_token_allowed(key, tokens=50, limit=100, window=1) is True
        )
        assert (
            await limiter.is_token_allowed(key, tokens=50, limit=100, window=1) is True
        )
        assert (
            await limiter.is_token_allowed(key, tokens=1, limit=100, window=1) is False
        )
        assert mock_redis.pipeline.return_value.zremrangebyscore.call_count == 3
        assert mock_redis.pipeline.return_value.zrange.call_count == 3
        # zadd for tokens is called outside pipeline in the current implementation
        assert (
            mock_redis.zadd.call_count == 2
        )  # Called for the first two successful calls


@pytest.mark.asyncio
class TestRateLimiterDispatch:
    @patch("open_reranker.core.rate_limiting.settings")
    async def test_uses_in_memory_if_no_redis_url(self, mock_rl_settings):
        mock_rl_settings.REDIS_URL = None
        limiter = RateLimiter()
        assert isinstance(limiter.memory_limiter, InMemoryRateLimiter)
        assert limiter.redis_limiter is None
        with patch.object(
            limiter.memory_limiter, "is_allowed", new_callable=AsyncMock
        ) as mock_mem_is_allowed:
            await limiter.check_rate_limit("test", 1, 1)
            mock_mem_is_allowed.assert_called_once()

    @patch("open_reranker.core.rate_limiting.settings")
    @patch("open_reranker.core.rate_limiting.RedisRateLimiter")
    async def test_uses_redis_if_url_provided(
        self, MockRedisRateLimiter, mock_rl_settings
    ):
        mock_rl_settings.REDIS_URL = "redis://localhost"
        mock_redis_instance = AsyncMock()
        MockRedisRateLimiter.return_value = mock_redis_instance

        limiter = RateLimiter()
        assert limiter.redis_limiter == mock_redis_instance

        await limiter.check_rate_limit("test", 1, 1)
        mock_redis_instance.is_allowed.assert_called_once()

    @patch("open_reranker.core.rate_limiting.settings")
    @patch(
        "open_reranker.core.rate_limiting.RedisRateLimiter",
        side_effect=Exception("Redis connect error"),
    )
    async def test_falls_back_to_in_memory_on_redis_error(
        self, MockRedisRateLimiter, mock_rl_settings
    ):
        mock_rl_settings.REDIS_URL = "redis://localhost"
        limiter = RateLimiter()
        assert isinstance(limiter.memory_limiter, InMemoryRateLimiter)
        assert limiter.redis_limiter is None  # Should be None after failing to connect
        with patch.object(
            limiter.memory_limiter, "is_allowed", new_callable=AsyncMock
        ) as mock_mem_is_allowed:
            await limiter.check_rate_limit("test", 1, 1)
            mock_mem_is_allowed.assert_called_once()


def test_get_client_id_ip(mock_request):
    assert get_client_id(mock_request) == "ip:127.0.0.1"


def test_get_client_id_ip_forwarded(mock_request):
    mock_request.headers = {"X-Forwarded-For": "1.2.3.4, 5.6.7.8"}
    assert get_client_id(mock_request) == "ip:1.2.3.4"


def test_get_client_id_user(mock_request, mock_user):
    assert get_client_id(mock_request, mock_user) == "user:test_user"


@pytest.mark.asyncio
async def test_check_rate_limits_qps_exceeded(
    mock_settings_rate_limit, mock_request, mock_user
):
    with patch("open_reranker.core.rate_limiting.settings", mock_settings_rate_limit):
        limiter = RateLimiter()
        with patch(
            "open_reranker.core.rate_limiting.rate_limiter", limiter
        ):  # Patch global instance
            # First 2 calls should pass for QPS=2
            await check_rate_limits(mock_request, mock_user, 10)
            await check_rate_limits(mock_request, mock_user, 10)
            # Third call should fail
            with pytest.raises(HTTPException) as exc_info:
                await check_rate_limits(mock_request, mock_user, 10)
            assert exc_info.value.status_code == 429
            assert "requests per second" in exc_info.value.detail


@pytest.mark.asyncio
async def test_check_rate_limits_rpm_exceeded(
    mock_settings_rate_limit, mock_request, mock_user
):
    mock_settings_rate_limit.RATE_LIMIT_QPS = 10  # Avoid QPS limit for this test
    with patch("open_reranker.core.rate_limiting.settings", mock_settings_rate_limit):
        limiter = RateLimiter()  # Fresh limiter for this test
        with patch("open_reranker.core.rate_limiting.rate_limiter", limiter):
            for _ in range(5):  # RPM limit is 5
                await check_rate_limits(mock_request, mock_user, 1)
            with pytest.raises(HTTPException) as exc_info:
                await check_rate_limits(mock_request, mock_user, 1)
            assert exc_info.value.status_code == 429
            assert "requests per minute" in exc_info.value.detail


@pytest.mark.asyncio
async def test_check_rate_limits_tpm_exceeded(
    mock_settings_rate_limit, mock_request, mock_user
):
    mock_settings_rate_limit.RATE_LIMIT_QPS = 10
    mock_settings_rate_limit.RATE_LIMIT_RPM = 20  # Avoid other limits
    with patch("open_reranker.core.rate_limiting.settings", mock_settings_rate_limit):
        limiter = RateLimiter()  # Fresh limiter
        with patch("open_reranker.core.rate_limiting.rate_limiter", limiter):
            await check_rate_limits(mock_request, mock_user, token_count=50)
            await check_rate_limits(
                mock_request, mock_user, token_count=50
            )  # Total 100, limit 100
            with pytest.raises(HTTPException) as exc_info:
                await check_rate_limits(
                    mock_request, mock_user, token_count=1
                )  # Exceeds by 1
            assert exc_info.value.status_code == 429
            assert "tokens per minute" in exc_info.value.detail


@pytest.mark.asyncio
async def test_rate_limit_dependency_success(
    mock_settings_rate_limit, mock_request, mock_user
):
    with patch("open_reranker.core.rate_limiting.settings", mock_settings_rate_limit):
        limiter = RateLimiter()
        with patch("open_reranker.core.rate_limiting.rate_limiter", limiter):
            # Mock get_current_user_optional dependency within the rate_limit_dependency scope
            async def mock_get_user_opt(request):
                return mock_user

            dependency_func = rate_limit_dependency(token_count=10)
            # Manually resolve the inner dependency for testing
            with patch(
                "open_reranker.core.rate_limiting.get_current_user_optional",
                new=mock_get_user_opt,
            ):
                user_from_dep = await dependency_func(request=mock_request)
            assert user_from_dep == mock_user


@pytest.mark.asyncio
async def test_rate_limit_dependency_fail(
    mock_settings_rate_limit, mock_request, mock_user
):
    with patch("open_reranker.core.rate_limiting.settings", mock_settings_rate_limit):
        limiter = RateLimiter()
        # Pre-fill QPS limit
        client_id = get_client_id(mock_request, mock_user)
        await limiter.check_rate_limit(
            f"{client_id}:qps", mock_settings_rate_limit.RATE_LIMIT_QPS, 1
        )
        await limiter.check_rate_limit(
            f"{client_id}:qps", mock_settings_rate_limit.RATE_LIMIT_QPS, 1
        )

        with patch("open_reranker.core.rate_limiting.rate_limiter", limiter):

            async def mock_get_user_opt(request):
                return mock_user

            dependency_func = rate_limit_dependency(token_count=10)
            with patch(
                "open_reranker.core.rate_limiting.get_current_user_optional",
                new=mock_get_user_opt,
            ):
                with pytest.raises(HTTPException) as exc_info:
                    await dependency_func(request=mock_request)
                assert exc_info.value.status_code == 429

            async def mock_get_user_opt(request):
                return mock_user

            dependency_func = rate_limit_dependency(token_count=10)
            with patch(
                "open_reranker.core.rate_limiting.get_current_user_optional",
                new=mock_get_user_opt,
            ):
                with pytest.raises(HTTPException) as exc_info:
                    await dependency_func(request=mock_request)
                assert exc_info.value.status_code == 429
