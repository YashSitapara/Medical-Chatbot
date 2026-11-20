"""GET /ready: Redis ping + Pinecone stats, each with a time limit, run concurrently."""

import asyncio
import time

import fakeredis
import httpx
import pytest

from app.config import Settings
from app.main import create_app
from app.memory import InMemoryChatMemory, RedisChatMemory
from app.readiness import run_check
from tests.conftest import StubIndex, make_rag

TIMEOUT = 0.3


class SlowMemory(InMemoryChatMemory):
    """A memory backend whose ping takes `delay` seconds (a hung Redis)."""

    backend = "redis"

    def __init__(self, delay: float):
        super().__init__()
        self.delay = delay

    async def ping(self) -> None:
        await asyncio.sleep(self.delay)


@pytest.fixture
def ready_settings() -> Settings:
    return Settings(pinecone_api_key="test-key", llm_provider="ollama", ready_timeout_seconds=TIMEOUT)


def make_client(settings, memory=None, index=None, redis_up=True):
    rag, store = make_rag(settings, [])
    if index is not None:
        index.store = store
        store.index = index
    if memory is None:
        server = fakeredis.FakeServer()
        server.connected = redis_up
        memory = RedisChatMemory(client=fakeredis.FakeAsyncRedis(server=server, decode_responses=True))
    app = create_app()
    app.state.rag = rag
    app.state.memory = memory
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test"), store


async def test_ready_when_both_pass(ready_settings):
    client, store = make_client(ready_settings)
    r = await client.get("/ready")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ready"
    assert set(body["checks"]) == {"redis", "pinecone"}
    assert all(c["ok"] and c["error"] is None and c["latency_ms"] >= 0 for c in body["checks"].values())
    # The Pinecone call carries its own HTTP timeout, so a hung thread ends by itself.
    assert store.index.calls == [{"_request_timeout": TIMEOUT}]


async def test_redis_down_is_503_and_named(ready_settings):
    client, _ = make_client(ready_settings, redis_up=False)
    r = await client.get("/ready")
    assert r.status_code == 503
    checks = r.json()["checks"]
    assert r.json()["status"] == "not_ready"
    assert checks["redis"] == {**checks["redis"], "ok": False, "error": "MemoryUnavailableError"}
    assert checks["pinecone"]["ok"] is True


async def test_pinecone_error_is_503_without_leaking_details(ready_settings):
    index = StubIndex(error=RuntimeError("401 Unauthorized for key pcsk_SECRET at internal-host"))
    client, _ = make_client(ready_settings, index=index)
    r = await client.get("/ready")
    assert r.status_code == 503
    checks = r.json()["checks"]
    assert checks["redis"]["ok"] is True
    assert checks["pinecone"]["ok"] is False and checks["pinecone"]["error"] == "RuntimeError"
    assert "SECRET" not in r.text and "internal-host" not in r.text


async def test_pinecone_timeout_is_503(ready_settings):
    client, _ = make_client(ready_settings, index=StubIndex(delay=TIMEOUT + 0.3))
    start = time.perf_counter()
    r = await client.get("/ready")
    assert time.perf_counter() - start < TIMEOUT + 0.25  # did not wait for the slow call
    assert r.status_code == 503
    assert r.json()["checks"]["pinecone"]["error"] == f"timed out after {TIMEOUT:g}s"
    assert r.json()["checks"]["redis"]["ok"] is True


async def test_redis_timeout_is_503(ready_settings):
    client, _ = make_client(ready_settings, memory=SlowMemory(delay=5))
    r = await client.get("/ready")
    assert r.status_code == 503
    assert r.json()["checks"]["redis"]["error"] == f"timed out after {TIMEOUT:g}s"


async def test_both_failing_are_both_reported(ready_settings):
    client, _ = make_client(ready_settings, redis_up=False, index=StubIndex(error=ConnectionError("x")))
    r = await client.get("/ready")
    assert r.status_code == 503
    assert {k: v["ok"] for k, v in r.json()["checks"].items()} == {"redis": False, "pinecone": False}


async def test_checks_run_concurrently(ready_settings):
    """Two checks of ~0.2s each must take ~0.2s in total, not ~0.4s."""
    client, _ = make_client(ready_settings, memory=SlowMemory(delay=0.2), index=StubIndex(delay=0.2))
    start = time.perf_counter()
    r = await client.get("/ready")
    elapsed = time.perf_counter() - start
    assert r.status_code == 200
    assert elapsed < 0.35


async def test_inmemory_backend_check_is_named_memory(ready_settings):
    client, _ = make_client(ready_settings, memory=InMemoryChatMemory())
    r = await client.get("/ready")
    assert r.status_code == 200
    assert set(r.json()["checks"]) == {"memory", "pinecone"}


# --- run_check on its own ------------------------------------------------------------


async def test_run_check_success():
    async def ok():
        return None

    result = await run_check("x", ok, 1)
    assert result.ok and result.error is None


async def test_run_check_exception():
    async def boom():
        raise ValueError("details")

    result = await run_check("x", boom, 1)
    assert not result.ok and result.error == "ValueError"


async def test_run_check_timeout():
    result = await run_check("x", lambda: asyncio.sleep(1), 0.05)
    assert not result.ok and result.error == "timed out after 0.05s"
    assert result.latency_ms < 500
