"""Memory tests.

The contract tests run against every backend (in-memory, fakeredis, and a real Redis
when REDIS_URL is set) so both implementations are guaranteed to behave the same.
"""

import asyncio
import os
import uuid

import fakeredis
import pytest
from langchain_core.messages import AIMessage, HumanMessage
from redis.exceptions import ConnectionError as RedisConnectionError

from app.config import Settings
from app.memory import (
    InMemoryChatMemory,
    MemoryUnavailableError,
    RedisChatMemory,
    build_memory,
    redact_url,
    to_langchain,
)

MAX_TURNS = 3


@pytest.fixture(
    params=[
        "memory",
        "fakeredis",
        pytest.param("real-redis", marks=pytest.mark.redis),
    ]
)
async def memory(request):
    if request.param == "memory":
        yield InMemoryChatMemory(max_turns=MAX_TURNS)
        return
    if request.param == "fakeredis":
        client = fakeredis.FakeAsyncRedis(decode_responses=True)
        mem = RedisChatMemory(max_turns=MAX_TURNS, client=client)
    else:
        url = os.environ.get("REDIS_URL")
        if not url:
            pytest.skip("set REDIS_URL to run against a real Redis")
        # Unique prefix so tests never touch real data.
        mem = RedisChatMemory(url, max_turns=MAX_TURNS, key_prefix=f"test:{uuid.uuid4().hex}:")
    yield mem
    if request.param == "real-redis":  # leave the real server exactly as we found it
        keys = [k async for k in mem._redis.scan_iter(match=f"{mem.key('')}*")]
        if keys:
            await mem._redis.delete(*keys)
    await mem.close()


# --- Contract: every backend ----------------------------------------------------


async def test_append_and_get_round_trip(memory):
    await memory.append("s1", "hi", "hello")
    msgs = await memory.get("s1")
    assert [(m.role, m.content) for m in msgs] == [("user", "hi"), ("assistant", "hello")]
    assert msgs[0].created_at.tzinfo is not None  # timestamps survive serialisation


async def test_unknown_session(memory):
    assert await memory.get("nope") == []
    assert not await memory.exists("nope")


async def test_turn_cap_drops_oldest(memory):
    for i in range(MAX_TURNS + 2):
        await memory.append("s", f"q{i}", f"a{i}")
    contents = [m.content for m in await memory.get("s")]
    assert contents == ["q2", "a2", "q3", "a3", "q4", "a4"]


async def test_sessions_are_isolated_and_clear(memory):
    await memory.append("a", "qa", "aa")
    await memory.append("b", "qb", "ab")
    assert await memory.clear("a") is True
    assert await memory.clear("a") is False
    assert await memory.get("a") == []
    assert not await memory.exists("a")
    assert len(await memory.get("b")) == 2


async def test_unicode_content(memory):
    await memory.append("u", "Qu’est-ce que l’acné ? 🤒", "Acne… [1]")
    assert [m.content for m in await memory.get("u")] == ["Qu’est-ce que l’acné ? 🤒", "Acne… [1]"]


async def test_ping(memory):
    await memory.ping()


# --- In-memory specific -----------------------------------------------------------


class FakeClock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t


async def test_inmemory_ttl_expiry_and_refresh():
    clock = FakeClock()
    mem = InMemoryChatMemory(ttl_seconds=60, clock=clock)
    await mem.append("s", "q", "a")
    clock.t = 50
    await mem.get("s")  # get() refreshes the TTL
    clock.t = 100
    assert await mem.exists("s")  # 50s since last get; exists() itself does not refresh
    clock.t = 200
    assert not await mem.exists("s")
    assert await mem.get("s") == []


# --- Redis specific ------------------------------------------------------------------


@pytest.fixture
async def redis_pair():
    client = fakeredis.FakeAsyncRedis(decode_responses=True)
    mem = RedisChatMemory(max_turns=MAX_TURNS, ttl_seconds=600, key_prefix="pfx:", client=client)
    yield mem, client
    await client.aclose()


async def test_redis_key_prefix_and_json_storage(redis_pair):
    mem, client = redis_pair
    await mem.append("abc", "q", "a")
    assert await client.keys("*") == ["pfx:abc"]
    raw = await client.lrange("pfx:abc", 0, -1)
    assert '"role":"user"' in raw[0] and '"content":"q"' in raw[0]


async def test_redis_ttl_set_and_refreshed_by_get_only(redis_pair):
    mem, client = redis_pair
    await mem.append("s", "q", "a")
    assert 0 < await client.ttl("pfx:s") <= 600

    await client.expire("pfx:s", 5)  # simulate time passing
    await mem.exists("s")
    assert await client.ttl("pfx:s") <= 5  # exists() does not refresh
    await mem.get("s")
    assert await client.ttl("pfx:s") > 5  # get() does


async def test_redis_get_does_not_create_key(redis_pair):
    mem, client = redis_pair
    assert await mem.get("ghost") == []
    assert await client.exists("pfx:ghost") == 0


async def test_redis_concurrent_appends_keep_turns_intact(redis_pair):
    """Many writers on one session (e.g. several workers) never split a user/assistant pair."""
    mem, _ = redis_pair
    await asyncio.gather(*(mem.append("s", f"q{i}", f"a{i}") for i in range(20)))
    msgs = await mem.get("s")
    assert len(msgs) == MAX_TURNS * 2
    for user, assistant in zip(msgs[::2], msgs[1::2]):
        assert user.role == "user" and assistant.role == "assistant"
        assert user.content[1:] == assistant.content[1:]  # q7 pairs with a7


async def test_redis_errors_become_memory_unavailable():
    server = fakeredis.FakeServer()
    server.connected = False  # simulate Redis being down
    mem = RedisChatMemory(client=fakeredis.FakeAsyncRedis(server=server, decode_responses=True))
    for call in (mem.get("s"), mem.append("s", "q", "a"), mem.exists("s"), mem.clear("s"), mem.ping()):
        with pytest.raises(MemoryUnavailableError):
            await call


def test_redact_url_hides_password():
    assert redact_url("redis://:s3cret@redis:6379/0") == "redis://:***@redis:6379/0"
    assert redact_url("rediss://user:s3cret@host:6380/1") == "rediss://user:***@host:6380/1"
    assert redact_url("redis://localhost:6379/0") == "redis://localhost:6379/0"


def test_unavailable_error_message_never_contains_password():
    mem = RedisChatMemory("redis://:s3cret@nowhere:6379/0", client=fakeredis.FakeAsyncRedis())
    assert "s3cret" not in str(mem._unavailable(RedisConnectionError("boom")))


# --- Factory + helpers ---------------------------------------------------------------


def test_build_memory_selects_backend():
    assert isinstance(build_memory(Settings(pinecone_api_key="k")), InMemoryChatMemory)
    redis_mem = build_memory(Settings(pinecone_api_key="k", memory_backend="redis", redis_url="redis://x:6379/0"))
    assert isinstance(redis_mem, RedisChatMemory) and redis_mem.backend == "redis"


async def test_to_langchain():
    mem = InMemoryChatMemory()
    await mem.append("s", "q", "a")
    out = to_langchain(await mem.get("s"))
    assert isinstance(out[0], HumanMessage) and isinstance(out[1], AIMessage)


def test_redis_client_retries_once_on_stale_connection():
    """Regression: first request after a Redis restart failed on a stale pooled connection."""
    mem = RedisChatMemory("redis://localhost:6379/0")
    retry = mem._redis.connection_pool.connection_kwargs["retry"]
    assert retry._retries == 1
    assert {e.__name__ for e in retry._supported_errors} >= {"ConnectionError", "TimeoutError"}
