"""Fixed-window rate limiting for the chat endpoints.

Each request increments up to two counters for the current minute:

    ratelimit:session:{session_id}:{minute}   limit RATE_LIMIT_PER_MINUTE        (if a session is sent)
    ratelimit:ip:{ip}:{minute}                limit RATE_LIMIT_PER_IP_PER_MINUTE

The IP counter matters because session_id comes from the client: a script could send a new
one with every request. Over either limit -> 429 with Retry-After.

Fixed window trade-off: up to 2x the limit can pass around a minute boundary (10 at 0:59,
10 at 1:00). Accepted for one INCR per counter and constant memory.

Fail open: if Redis is unreachable the request is ALLOWED and a warning is logged. The
limiter protects cost, it is not a security boundary; failing closed would turn a Redis
blip into a full chat outage.
"""

import math
import threading
import time
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass

import structlog

from app.config import Settings
from app.memory import redact_url

logger = structlog.get_logger(__name__)

WINDOW_SECONDS = 60
UNAVAILABLE_LOG_INTERVAL = 30.0  # seconds between "rate limiter down" warnings, per process


class RateLimiterUnavailable(Exception):
    pass


@dataclass
class RateLimitDecision:
    allowed: bool
    limit: int  # the limit of the counter closest to (or over) its limit
    remaining: int | None  # None when the backend was unavailable (fail open)
    retry_after: int  # seconds until the window resets
    limited_by: str | None = None  # "session" | "ip" when blocked

    def headers(self) -> dict[str, str]:
        headers = {"X-RateLimit-Limit": str(self.limit)}
        if self.remaining is not None:
            headers["X-RateLimit-Remaining"] = str(self.remaining)
        if not self.allowed:
            headers["Retry-After"] = str(self.retry_after)
        return headers


class FixedWindowRateLimiter(ABC):
    def __init__(self, per_session: int, per_ip: int, clock: Callable[[], float] = time.time) -> None:
        self.per_session = per_session
        self.per_ip = per_ip
        self._clock = clock
        self._last_unavailable_log = -math.inf

    @abstractmethod
    async def _increment(self, keys: list[str]) -> list[int]:
        """Atomically INCR each key (expiring at the end of its window). Return the new counts."""

    async def close(self) -> None:
        pass

    async def check(self, session_id: str | None, ip: str) -> RateLimitDecision:
        now = self._clock()
        window = int(now // WINDOW_SECONDS)
        retry_after = max(1, math.ceil(WINDOW_SECONDS - now % WINDOW_SECONDS))

        buckets: list[tuple[str, str, int]] = []
        if session_id:
            buckets.append(("session", f"ratelimit:session:{session_id}:{window}", self.per_session))
        buckets.append(("ip", f"ratelimit:ip:{ip}:{window}", self.per_ip))

        try:
            counts = await self._increment([key for _, key, _ in buckets])
        except RateLimiterUnavailable as exc:
            self._warn_unavailable(exc)
            tightest = min(limit for _, _, limit in buckets)
            return RateLimitDecision(allowed=True, limit=tightest, remaining=None, retry_after=retry_after)

        for (kind, _, limit), count in zip(buckets, counts):
            if count > limit:
                return RateLimitDecision(False, limit, 0, retry_after, limited_by=kind)
        # Report the counter closest to its limit: that's the one the client will hit first.
        limit, remaining = min(((lim, lim - c) for (_, _, lim), c in zip(buckets, counts)), key=lambda x: x[1])
        return RateLimitDecision(True, limit, remaining, retry_after)

    def _warn_unavailable(self, exc: Exception) -> None:
        now = time.monotonic()
        if now - self._last_unavailable_log >= UNAVAILABLE_LOG_INTERVAL:
            self._last_unavailable_log = now
            logger.warning("rate_limit_unavailable", detail=str(exc), action="allowing requests (fail open)")


class RedisRateLimiter(FixedWindowRateLimiter):
    """Shared by every worker and container."""

    def __init__(self, redis_url: str, per_session: int, per_ip: int, client=None, clock=time.time) -> None:
        super().__init__(per_session, per_ip, clock)
        from redis.asyncio import Redis

        self._url = redact_url(redis_url)
        # Own client with SHORT timeouts and no retry: a slow Redis may add at most ~0.5 s to a
        # chat request (the memory client allows 3 s + a retry, which is right for history).
        self._redis = client or Redis.from_url(
            redis_url, decode_responses=True, socket_connect_timeout=0.5, socket_timeout=0.5
        )

    async def _increment(self, keys: list[str]) -> list[int]:
        from redis.exceptions import RedisError

        try:
            async with self._redis.pipeline(transaction=True) as pipe:
                for key in keys:
                    pipe.incr(key)
                    pipe.expire(key, WINDOW_SECONDS, nx=True)  # only on the first hit (Redis >= 7)
                results = await pipe.execute()
        except (RedisError, OSError) as exc:
            raise RateLimiterUnavailable(f"Redis at {self._url}: {type(exc).__name__}") from exc
        return [int(n) for n in results[0::2]]

    async def close(self) -> None:
        await self._redis.aclose()


class InMemoryRateLimiter(FixedWindowRateLimiter):
    """Per-process counters, for MEMORY_BACKEND=memory (no Redis). Not shared between workers."""

    def __init__(self, per_session: int, per_ip: int, clock=time.time) -> None:
        super().__init__(per_session, per_ip, clock)
        self._counts: dict[str, int] = {}
        self._window = -1
        self._lock = threading.Lock()

    async def _increment(self, keys: list[str]) -> list[int]:
        window = int(self._clock() // WINDOW_SECONDS)
        with self._lock:
            if window != self._window:  # new minute: old counters can never match again
                self._counts.clear()
                self._window = window
            for key in keys:
                self._counts[key] = self._counts.get(key, 0) + 1
            return [self._counts[key] for key in keys]


def build_rate_limiter(settings: Settings) -> FixedWindowRateLimiter | None:
    if not settings.rate_limit_enabled:
        return None
    if settings.memory_backend == "redis":
        return RedisRateLimiter(settings.redis_url, settings.rate_limit_per_minute, settings.rate_limit_per_ip_per_minute)
    return InMemoryRateLimiter(settings.rate_limit_per_minute, settings.rate_limit_per_ip_per_minute)
