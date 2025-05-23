import time
from collections import defaultdict, deque
from typing import Any, Dict, Optional

import redis.asyncio as redis
from fastapi import Depends, HTTPException, Request

from open_reranker.core.config import settings
from open_reranker.core.logging import setup_logging
from open_reranker.core.auth import get_current_user_optional

logger = setup_logging()


class InMemoryRateLimiter:
    """In-memory rate limiter using sliding window."""

    def __init__(self):
        self.requests: Dict[str, deque] = defaultdict(deque)
        self.tokens: Dict[str, deque] = defaultdict(deque)

    async def is_allowed(self, key: str, limit: int, window: int) -> bool:
        """Check if request is allowed under rate limit."""
        now = time.time()
        window_start = now - window

        # Clean old requests
        while self.requests[key] and self.requests[key][0] < window_start:
            self.requests[key].popleft()

        # Check if under limit
        if len(self.requests[key]) >= limit:
            return False

        # Add current request
        self.requests[key].append(now)
        return True

    async def is_token_allowed(
        self, key: str, tokens: int, limit: int, window: int
    ) -> bool:
        """Check if token usage is allowed under rate limit."""
        now = time.time()
        window_start = now - window

        # Clean old token usage
        while self.tokens[key] and self.tokens[key][0][0] < window_start:
            self.tokens[key].popleft()

        # Calculate current token usage
        current_tokens = sum(token_count for _, token_count in self.tokens[key])

        # Check if under limit
        if current_tokens + tokens > limit:
            return False

        # Add current token usage
        self.tokens[key].append((now, tokens))
        return True


class RedisRateLimiter:
    """Redis-based rate limiter using sliding window."""

    def __init__(self, redis_url: str):
        self.redis = redis.from_url(redis_url)

    async def is_allowed(self, key: str, limit: int, window: int) -> bool:
        """Check if request is allowed under rate limit."""
        now = time.time()
        window_start = now - window

        pipe = self.redis.pipeline()
        pipe.zremrangebyscore(key, 0, window_start)
        pipe.zcard(key)
        pipe.zadd(key, {str(now): now})
        pipe.expire(key, window)

        results = await pipe.execute()
        current_requests = results[1]

        return current_requests < limit

    async def is_token_allowed(
        self, key: str, tokens: int, limit: int, window: int
    ) -> bool:
        """Check if token usage is allowed under rate limit."""
        token_key = f"{key}:tokens"
        now = time.time()
        window_start = now - window

        pipe = self.redis.pipeline()
        pipe.zremrangebyscore(token_key, 0, window_start)
        pipe.zrange(token_key, 0, -1, withscores=True)

        results = await pipe.execute()
        token_entries = results[1]

        current_tokens = sum(int(entry[0]) for entry in token_entries)

        if current_tokens + tokens > limit:
            return False

        # Add current token usage
        await self.redis.zadd(token_key, {str(tokens): now})
        await self.redis.expire(token_key, window)

        return True


class RateLimiter:
    """Main rate limiter that uses Redis if available, otherwise in-memory."""

    def __init__(self):
        self.redis_limiter = None
        self.memory_limiter = InMemoryRateLimiter()

        if settings.REDIS_URL:
            try:
                self.redis_limiter = RedisRateLimiter(settings.REDIS_URL)
                logger.info("Using Redis for rate limiting")
            except Exception as e:
                logger.warning(
                    f"Failed to connect to Redis, using in-memory rate limiting: {e}"
                )
        else:
            logger.info("Using in-memory rate limiting")

    async def check_rate_limit(self, key: str, limit: int, window: int = 60) -> bool:
        """Check if request is within rate limit."""
        limiter = self.redis_limiter if self.redis_limiter else self.memory_limiter
        return await limiter.is_allowed(key, limit, window)

    async def check_token_limit(
        self, key: str, tokens: int, limit: int, window: int = 60
    ) -> bool:
        """Check if token usage is within rate limit."""
        limiter = self.redis_limiter if self.redis_limiter else self.memory_limiter
        return await limiter.is_token_allowed(key, tokens, limit, window)


# Global rate limiter instance
rate_limiter = RateLimiter()


def get_client_id(request: Request, user: Optional[Dict[str, Any]] = None) -> str:
    """Get client identifier for rate limiting."""
    if user and user.get("sub"):
        return f"user:{user['sub']}"

    # Use IP address as fallback
    client_ip = request.client.host if request.client else "unknown"
    forwarded_for = request.headers.get("X-Forwarded-For")
    if forwarded_for:
        client_ip = forwarded_for.split(",")[0].strip()

    return f"ip:{client_ip}"


async def check_rate_limits(
    request: Request, user: Optional[Dict[str, Any]] = None, token_count: int = 0
) -> None:
    """Check all applicable rate limits."""
    client_id = get_client_id(request, user)
    user_tier = user.get("tier", "free") if user else "free"

    # Get limits based on user tier
    if user_tier == "enterprise":
        qps_limit = settings.RATE_LIMIT_QPS * 10
        rpm_limit = settings.RATE_LIMIT_RPM * 10
        tpm_limit = settings.RATE_LIMIT_TPM * 10
    elif user_tier == "pro":
        qps_limit = settings.RATE_LIMIT_QPS * 5
        rpm_limit = settings.RATE_LIMIT_RPM * 5
        tpm_limit = settings.RATE_LIMIT_TPM * 5
    else:  # free tier
        qps_limit = settings.RATE_LIMIT_QPS
        rpm_limit = settings.RATE_LIMIT_RPM
        tpm_limit = settings.RATE_LIMIT_TPM

    # Check QPS limit (1 second window)
    if not await rate_limiter.check_rate_limit(f"{client_id}:qps", qps_limit, 1):
        raise HTTPException(
            status_code=429,
            detail=f"Rate limit exceeded: {qps_limit} requests per second",
        )

    # Check RPM limit (60 second window)
    if not await rate_limiter.check_rate_limit(f"{client_id}:rpm", rpm_limit, 60):
        raise HTTPException(
            status_code=429,
            detail=f"Rate limit exceeded: {rpm_limit} requests per minute",
        )

    # Check TPM limit if token counting is enabled
    if settings.COUNT_TOKENS and token_count > 0:
        if not await rate_limiter.check_token_limit(
            f"{client_id}:tpm", token_count, tpm_limit, 60
        ):
            raise HTTPException(
                status_code=429,
                detail=f"Rate limit exceeded: {tpm_limit} tokens per minute",
            )


def rate_limit_dependency(token_count: int = 0):
    """Dependency function for rate limiting."""
    async def _rate_limit_dependency_implementation(
        request: Request,
        user: Optional[Dict[str, Any]] = Depends(get_current_user_optional)
    ):
        await check_rate_limits(request, user, token_count=token_count)
        return user
    return _rate_limit_dependency_implementation
