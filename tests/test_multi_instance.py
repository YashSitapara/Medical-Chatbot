"""Several app instances sharing one Redis - the multi-worker / multi-container scenario.

Each `create_app()` below is a separate app with its own RAG service and its own Redis
client, exactly like two uvicorn workers or two containers. They only share the Redis
*server*, so any history visible across them must have gone through Redis.
"""

import fakeredis
import httpx
import pytest

from app.main import create_app
from app.memory import InMemoryChatMemory, MemoryUnavailableError, RedisChatMemory
from tests.conftest import make_rag
from tests.test_api import parse_sse


def make_instance(settings, responses, server):
    rag, store = make_rag(settings, responses)
    app = create_app()
    app.state.rag = rag
    app.state.memory = RedisChatMemory(client=fakeredis.FakeAsyncRedis(server=server, decode_responses=True))
    client = httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test")
    return client, store


@pytest.fixture
def server():
    return fakeredis.FakeServer()


async def test_followup_on_another_instance_sees_history(settings, server):
    worker_a, store_a = make_instance(settings, ["Acne is a skin disease [1]."], server)
    worker_b, store_b = make_instance(settings, ["What causes acne?", "Clogged pores [2]."], server)

    sid = (await worker_a.post("/api/chat", json={"message": "What is acne?"})).json()["session_id"]

    # The follow-up lands on a different worker: it must still be rewritten using A's turn.
    r = await worker_b.post("/api/chat", json={"message": "What causes it?", "session_id": sid})
    assert r.json()["answer"] == "Clogged pores [2]."
    assert store_a.queries == ["What is acne?"]
    assert store_b.queries == ["What causes acne?"]

    history = (await worker_a.get(f"/api/sessions/{sid}/history")).json()["messages"]
    assert [m["content"] for m in history] == [
        "What is acne?", "Acne is a skin disease [1].", "What causes it?", "Clogged pores [2].",
    ]


async def test_streamed_turn_is_visible_to_other_instance(settings, server):
    worker_a, _ = make_instance(settings, ["Acne [1]."], server)
    worker_b, _ = make_instance(settings, [], server)

    r = await worker_a.post("/api/chat/stream", json={"message": "What is acne?"})
    events = parse_sse(r.text)
    sid = events[0][1]["session_id"]
    assert events[-1][0] == "done"

    history = (await worker_b.get(f"/api/sessions/{sid}/history")).json()["messages"]
    assert [m["content"] for m in history] == ["What is acne?", "Acne [1]."]


async def test_new_chat_on_one_instance_clears_everywhere(settings, server):
    worker_a, _ = make_instance(settings, ["A [1]."], server)
    worker_b, _ = make_instance(settings, [], server)

    sid = (await worker_a.post("/api/chat", json={"message": "q"})).json()["session_id"]
    assert (await worker_b.delete(f"/api/sessions/{sid}")).status_code == 204
    assert (await worker_a.get(f"/api/sessions/{sid}/history")).status_code == 404


async def test_inmemory_backend_does_not_share(settings):
    """Documents the bug Redis fixes: separate in-memory instances can't see each other."""
    apps = []
    for responses in (["A [1]."], []):
        rag, _ = make_rag(settings, responses)
        app = create_app()
        app.state.rag, app.state.memory = rag, InMemoryChatMemory()
        apps.append(httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test"))

    sid = (await apps[0].post("/api/chat", json={"message": "q"})).json()["session_id"]
    assert (await apps[1].get(f"/api/sessions/{sid}/history")).status_code == 404


# --- Redis down ----------------------------------------------------------------------


@pytest.fixture
def down_instance(settings):
    server = fakeredis.FakeServer()
    client, _ = make_instance(settings, ["unused"], server)
    server.connected = False
    return client


async def test_chat_returns_503_when_redis_down(down_instance):
    r = await down_instance.post("/api/chat", json={"message": "q"})
    assert r.status_code == 503
    body = r.json()
    assert body["detail"] == "Chat memory is temporarily unavailable."
    assert body["code"] == "dependency_unavailable" and body["request_id"] == r.headers["X-Request-ID"]


async def test_stream_returns_503_before_streaming_when_redis_down(down_instance):
    r = await down_instance.post("/api/chat/stream", json={"message": "q"})
    assert r.status_code == 503


async def test_history_and_health_return_503_when_redis_down(down_instance):
    assert (await down_instance.get("/api/sessions/x/history")).status_code == 503
    assert (await down_instance.get("/health")).status_code == 503


async def test_health_reports_redis_backend(settings, server):
    client, _ = make_instance(settings, [], server)
    body = (await client.get("/health")).json()
    assert body["status"] == "ok" and body["memory_backend"] == "redis"


async def test_startup_fails_fast_when_redis_unreachable(settings):
    server = fakeredis.FakeServer()
    server.connected = False
    rag, _ = make_rag(settings, [])
    app = create_app()
    app.state.rag = rag
    app.state.memory = RedisChatMemory(client=fakeredis.FakeAsyncRedis(server=server, decode_responses=True))
    with pytest.raises(MemoryUnavailableError):
        async with app.router.lifespan_context(app):
            pass


async def test_warns_when_inmemory_used_with_multiple_workers(settings, monkeypatch, caplog):
    monkeypatch.setenv("WEB_CONCURRENCY", "3")
    rag, _ = make_rag(settings, [])
    app = create_app()
    app.state.rag, app.state.memory = rag, InMemoryChatMemory()
    with caplog.at_level("WARNING", logger="medical_rag"):
        async with app.router.lifespan_context(app):
            pass
    assert "Set MEMORY_BACKEND=redis" in caplog.text


async def test_no_warning_for_redis_with_multiple_workers(settings, monkeypatch, caplog):
    monkeypatch.setenv("WEB_CONCURRENCY", "3")
    rag, _ = make_rag(settings, [])
    app = create_app()
    app.state.rag = rag
    app.state.memory = RedisChatMemory(client=fakeredis.FakeAsyncRedis(decode_responses=True))
    with caplog.at_level("WARNING", logger="medical_rag"):
        async with app.router.lifespan_context(app):
            pass
    assert "MEMORY_BACKEND" not in caplog.text
