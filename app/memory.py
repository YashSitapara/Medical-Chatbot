"""Per-session chat history.

`ChatMemory` is the async interface the rest of the app depends on. Two backends:

* `InMemoryChatMemory` - a dict inside this process. Zero setup, but lost on restart
  and NOT shared between uvicorn workers or containers (use a single worker).
* `RedisChatMemory` - one Redis list per session. Shared by every worker and
  container, survives app restarts; Redis handles size limits and expiry natively.

The interface is async because a Redis call is network I/O: a synchronous call inside
an `async` route would block the event loop for every other request in that worker.
"""

import threading
import time
from abc import ABC, abstractmethod
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from urllib.parse import urlsplit, urlunsplit

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage

from app.config import Settings
from app.errors import DependencyUnavailableError
from app.schemas import Message


class MemoryUnavailableError(DependencyUnavailableError):
    """The memory backend can't be reached. Mapped to HTTP 503 by the API."""

    public_detail = "Chat memory is temporarily unavailable."


class ChatMemory(ABC):
    backend: str = "unknown"

    @abstractmethod
    async def get(self, session_id: str) -> list[Message]:
        """Return the session's messages (oldest first) and refresh its TTL."""

    @abstractmethod
    async def append(self, session_id: str, user: str, assistant: str, trace_id: str | None = None) -> None:
        """Store one completed turn (user question + assistant answer)."""

    @abstractmethod
    async def clear(self, session_id: str) -> bool:
        """Delete the session. Returns True if it existed."""

    @abstractmethod
    async def exists(self, session_id: str) -> bool:
        """Whether the session exists. Does NOT refresh its TTL."""

    async def ping(self) -> None:
        """Raise MemoryUnavailableError if the backend is unreachable."""

    async def close(self) -> None:
        """Release connections on shutdown."""


# ---------------------------------------------------------------------------
# In-process backend
# ---------------------------------------------------------------------------


@dataclass
class _Session:
    messages: deque[Message]
    last_access: float = field(default_factory=time.monotonic)


class InMemoryChatMemory(ChatMemory):
    backend = "memory"

    def __init__(
        self,
        max_turns: int = 10,
        ttl_seconds: int = 3600,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._max_messages = max_turns * 2  # one turn = user + assistant
        self._ttl = ttl_seconds
        self._clock = clock
        self._sessions: dict[str, _Session] = {}
        # Methods contain no `await`, so the lock is never held across a suspension.
        self._lock = threading.Lock()

    def _evict_expired(self) -> None:
        now = self._clock()
        expired = [sid for sid, s in self._sessions.items() if now - s.last_access > self._ttl]
        for sid in expired:
            del self._sessions[sid]

    async def get(self, session_id: str) -> list[Message]:
        with self._lock:
            self._evict_expired()
            session = self._sessions.get(session_id)
            if session is None:
                return []
            session.last_access = self._clock()
            return list(session.messages)

    async def append(self, session_id: str, user: str, assistant: str, trace_id: str | None = None) -> None:
        with self._lock:
            self._evict_expired()
            session = self._sessions.get(session_id)
            if session is None:
                session = _Session(messages=deque(maxlen=self._max_messages), last_access=self._clock())
                self._sessions[session_id] = session
            session.messages.append(Message(role="user", content=user))
            session.messages.append(Message(role="assistant", content=assistant, trace_id=trace_id))
            session.last_access = self._clock()

    async def clear(self, session_id: str) -> bool:
        with self._lock:
            return self._sessions.pop(session_id, None) is not None

    async def exists(self, session_id: str) -> bool:
        with self._lock:
            self._evict_expired()
            return session_id in self._sessions


# ---------------------------------------------------------------------------
# Redis backend
# ---------------------------------------------------------------------------


def redact_url(url: str) -> str:
    """Hide the password in a redis:// URL before logging it."""
    parts = urlsplit(url)
    if parts.password:
        netloc = parts.netloc.replace(f":{parts.password}@", ":***@")
        return urlunsplit(parts._replace(netloc=netloc))
    return url


class RedisChatMemory(ChatMemory):
    """Each session is a Redis LIST of JSON-encoded `Message`s at `<prefix><session_id>`.

    * append: MULTI / RPUSH user+assistant / LTRIM to the cap / EXPIRE / EXEC
      (a transaction, so concurrent writers from different workers can't interleave a turn)
    * get:    LRANGE + EXPIRE (sliding TTL, same rule as the in-memory backend)
    """

    backend = "redis"

    def __init__(
        self,
        redis_url: str = "redis://localhost:6379/0",
        max_turns: int = 10,
        ttl_seconds: int = 3600,
        key_prefix: str = "medical-rag:chat:",
        client=None,
    ) -> None:
        from redis.asyncio import Redis
        from redis.asyncio.retry import Retry
        from redis.backoff import ExponentialBackoff

        self._url = redact_url(redis_url)
        self._max_messages = max_turns * 2
        self._ttl = ttl_seconds
        self._prefix = key_prefix
        # One connection pool per worker process, shared by all its requests.
        self._redis: Redis = client or Redis.from_url(
            redis_url,
            decode_responses=True,
            socket_connect_timeout=3,
            socket_timeout=3,
            health_check_interval=30,
            # After a Redis restart, pooled connections point at the dead server and the
            # first command on each fails ("Connection closed by server"). One retry on a
            # fresh connection hides that; a real outage still fails fast (refused is instant).
            # Rare caveat: if a connection drops after EXEC ran but before its reply, a
            # retried append stores the turn twice (LTRIM still bounds the list).
            retry=Retry(ExponentialBackoff(cap=0.5, base=0.05), retries=1),
        )

    def key(self, session_id: str) -> str:
        return f"{self._prefix}{session_id}"

    def _unavailable(self, exc: Exception) -> MemoryUnavailableError:
        return MemoryUnavailableError(f"Redis at {self._url} is unavailable: {exc}")

    async def get(self, session_id: str) -> list[Message]:
        from redis.exceptions import RedisError

        key = self.key(session_id)
        try:
            async with self._redis.pipeline(transaction=True) as pipe:
                pipe.lrange(key, 0, -1)
                pipe.expire(key, self._ttl)  # no-op if the key doesn't exist
                raw, _ = await pipe.execute()
        except RedisError as exc:
            raise self._unavailable(exc) from exc
        return [Message.model_validate_json(item) for item in raw]

    async def append(self, session_id: str, user: str, assistant: str, trace_id: str | None = None) -> None:
        from redis.exceptions import RedisError

        key = self.key(session_id)
        items = [
            Message(role="user", content=user).model_dump_json(),
            Message(role="assistant", content=assistant, trace_id=trace_id).model_dump_json(),
        ]
        try:
            async with self._redis.pipeline(transaction=True) as pipe:
                pipe.rpush(key, *items)
                pipe.ltrim(key, -self._max_messages, -1)  # keep only the newest N messages
                pipe.expire(key, self._ttl)
                await pipe.execute()
        except RedisError as exc:
            raise self._unavailable(exc) from exc

    async def clear(self, session_id: str) -> bool:
        from redis.exceptions import RedisError

        try:
            return bool(await self._redis.delete(self.key(session_id)))
        except RedisError as exc:
            raise self._unavailable(exc) from exc

    async def exists(self, session_id: str) -> bool:
        from redis.exceptions import RedisError

        try:
            return bool(await self._redis.exists(self.key(session_id)))
        except RedisError as exc:
            raise self._unavailable(exc) from exc

    async def ping(self) -> None:
        from redis.exceptions import RedisError

        try:
            await self._redis.ping()
        except (RedisError, OSError) as exc:
            raise self._unavailable(exc) from exc

    async def close(self) -> None:
        await self._redis.aclose()


# ---------------------------------------------------------------------------
# Factory + helpers
# ---------------------------------------------------------------------------


def build_memory(settings: Settings) -> ChatMemory:
    if settings.memory_backend == "redis":
        return RedisChatMemory(
            redis_url=settings.redis_url,
            max_turns=settings.memory_max_turns,
            ttl_seconds=settings.memory_ttl_seconds,
            key_prefix=settings.redis_key_prefix,
        )
    return InMemoryChatMemory(settings.memory_max_turns, settings.memory_ttl_seconds)


def to_langchain(messages: list[Message]) -> list[BaseMessage]:
    """Convert stored messages into LangChain chat messages for the prompt."""
    return [
        HumanMessage(content=m.content) if m.role == "user" else AIMessage(content=m.content)
        for m in messages
    ]


__all__ = [
    "ChatMemory",
    "InMemoryChatMemory",
    "MemoryUnavailableError",
    "RedisChatMemory",
    "build_memory",
    "redact_url",
    "to_langchain",
]
