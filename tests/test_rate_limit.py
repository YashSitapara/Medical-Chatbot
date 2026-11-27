"""Fixed-window rate limiting: per session + per IP, 429 + Retry-After, fail open."""

import asyncio
import os
import uuid

import fakeredis
import pytest
from fastapi.testclient import TestClient
from langchain_core.language_models.fake_chat_models import FakeListChatModel

from app.config import Settings
from app.main import create_app
from app.memory import InMemoryChatMemory
from app.rag import RAGService
from app.rate_limit import InMemoryRateLimiter, RedisRateLimiter, build_rate_limiter
from tests.conftest import StubVectorStore, SID, new_sid
from tests.test_api import parse_sse

WINDOW_START = 60 * 29_000_000  # an exact minute boundary
CLIENT_IP = "203.0.113.7"


class Clock:
    def __init__(self, t: float = WINDOW_START + 5):
        self.t = t

    def __call__(self) -> float:
        return self.t


def make_limiter(kind, clock, per_session=10, per_ip=30, redis_client=None):
    if kind == "memory":
        return InMemoryRateLimiter(per_session, per_ip, clock=clock)
    client = redis_client or fakeredis.FakeAsyncRedis(decode_responses=True)
    return RedisRateLimiter("redis://fake:6379/0", per_session, per_ip, client=client, clock=clock)


def make_client(limiter, ip=CLIENT_IP):
    settings = Settings(pinecone_api_key="test-key", llm_provider="ollama")
    rag = RAGService(settings, llm=FakeListChatModel(responses=["Acne [1]."]), vectorstore=StubVectorStore())
    app = create_app()
    app.state.rag = rag
    app.state.memory = InMemoryChatMemory()
    app.state.rate_limiter = limiter
    client = TestClient(app, client=(ip, 50000), raise_server_exceptions=False)
    client.__enter__()
    return client


def chat(client, session_id=SID, stream=False, **kw):
    body = {"message": "What is acne?", **({"session_id": session_id} if session_id else {})}
    return client.post("/api/chat/stream" if stream else "/api/chat", json=body, **kw)


@pytest.fixture(params=["redis", "memory"])
def kind(request):
    return request.param


# --- Core behaviour (both backends) -----------------------------------------------------


def test_eleventh_call_in_a_minute_is_429(kind):
    clock = Clock(WINDOW_START + 37)
    client = make_client(make_limiter(kind, clock))
    remaining = [chat(client).headers["X-RateLimit-Remaining"] for _ in range(10)]
    assert remaining == [str(n) for n in range(9, -1, -1)]

    r = chat(client)
    assert r.status_code == 429
    assert r.headers["Retry-After"] == "23"  # 60 - 37 seconds left in this window
    assert r.headers["X-RateLimit-Limit"] == "10" and r.headers["X-RateLimit-Remaining"] == "0"
    assert r.json() == {"detail": "Rate limit exceeded. Try again in 23 seconds.", "code": "rate_limited",
                        "request_id": r.headers["X-Request-ID"]}


def test_new_minute_starts_a_new_window(kind):
    clock = Clock()
    client = make_client(make_limiter(kind, clock, per_session=2))
    assert [chat(client).status_code for _ in range(3)] == [200, 200, 429]
    clock.t += 60
    assert chat(client).status_code == 200


def test_ip_cap_stops_rotating_session_ids(kind):
    client = make_client(make_limiter(kind, Clock(), per_session=10, per_ip=30))
    codes = [chat(client, session_id=new_sid()).status_code for _ in range(31)]
    assert codes == [200] * 30 + [429]


def test_headers_report_the_tightest_counter(kind):
    client = make_client(make_limiter(kind, Clock(), per_session=10, per_ip=3))
    r = chat(client)
    assert r.headers["X-RateLimit-Limit"] == "3" and r.headers["X-RateLimit-Remaining"] == "2"


def test_stream_route_is_limited_and_has_headers(kind):
    client = make_client(make_limiter(kind, Clock(), per_session=1))
    r = chat(client, stream=True)
    assert r.status_code == 200 and parse_sse(r.text)[-1][0] == "done"
    assert r.headers["X-RateLimit-Limit"] == "1" and r.headers["X-RateLimit-Remaining"] == "0"
    r = chat(client, stream=True)
    assert r.status_code == 429 and r.headers["Retry-After"]


def test_other_routes_are_never_limited(kind):
    client = make_client(make_limiter(kind, Clock(), per_session=1, per_ip=1))
    sid = chat(client).json()["session_id"]
    for _ in range(5):
        for r in (client.get("/health"), client.get(f"/api/sessions/{sid}/history"), client.get("/")):
            assert r.status_code == 200 and "X-RateLimit-Limit" not in r.headers


# --- Redis specifics ------------------------------------------------------------------------


def test_redis_keys_window_and_ttl_not_extended():
    redis = fakeredis.FakeAsyncRedis(decode_responses=True)
    client = make_client(make_limiter("redis", Clock(), redis_client=redis))
    window = (WINDOW_START + 5) // 60
    chat(client, session_id=None)  # no session: only the IP counter
    keys = asyncio.run(redis.keys("ratelimit:*"))
    assert keys == [f"ratelimit:ip:{CLIENT_IP}:{window}"]
    assert 0 < asyncio.run(redis.ttl(keys[0])) <= 60

    asyncio.run(redis.expire(keys[0], 20))  # time passes
    chat(client, session_id=None)
    assert asyncio.run(redis.ttl(keys[0])) <= 20  # EXPIRE ... NX: later hits don't reset it
    assert asyncio.run(redis.get(keys[0])) == "2"


def test_redis_down_fails_open_with_one_warning(caplog):
    server = fakeredis.FakeServer()
    server.connected = False
    redis = fakeredis.FakeAsyncRedis(server=server, decode_responses=True)
    client = make_client(make_limiter("redis", Clock(), per_session=1, redis_client=redis))
    responses = [chat(client) for _ in range(5)]
    assert [r.status_code for r in responses] == [200] * 5  # over the limit, but allowed
    assert responses[0].headers["X-RateLimit-Limit"] == "1"
    assert "X-RateLimit-Remaining" not in responses[0].headers  # unknown while Redis is down
    warnings = [r for r in caplog.records if isinstance(r.msg, dict) and r.msg["event"] == "rate_limit_unavailable"]
    assert len(warnings) == 1


def test_disabled_means_no_limit_and_no_headers():
    assert build_rate_limiter(Settings(pinecone_api_key="k", rate_limit_enabled=False)) is None
    client = make_client(None)
    for _ in range(15):
        r = chat(client)
        assert r.status_code == 200 and "X-RateLimit-Limit" not in r.headers


def test_backend_follows_memory_backend():
    on = {"pinecone_api_key": "k", "rate_limit_enabled": True}  # conftest disables it by default
    assert isinstance(build_rate_limiter(Settings(**on)), InMemoryRateLimiter)
    redis_limiter = build_rate_limiter(Settings(**on, memory_backend="redis"))
    assert isinstance(redis_limiter, RedisRateLimiter)


# --- Client identity and privacy ---------------------------------------------------------------


def test_forwarded_for_from_a_direct_client_is_ignored():
    """The app uses request.client.host; only uvicorn (with a trusted proxy) may rewrite it."""
    redis = fakeredis.FakeAsyncRedis(decode_responses=True)
    client = make_client(make_limiter("redis", Clock(), redis_client=redis), ip="198.51.100.9")
    chat(client, session_id=None, headers={"X-Forwarded-For": "1.2.3.4"})
    keys = asyncio.run(redis.keys("ratelimit:ip:*"))
    assert len(keys) == 1 and "198.51.100.9" in keys[0] and "1.2.3.4" not in keys[0]


def test_rate_limited_log_has_no_ip_or_counter_key(caplog):
    client = make_client(make_limiter("memory", Clock(), per_session=1))
    chat(client)
    chat(client)
    line = next(r.msg for r in caplog.records if isinstance(r.msg, dict) and r.msg["event"] == "rate_limited")
    assert line["limited_by"] == "session" and line["limit"] == 1
    assert CLIENT_IP not in str(line) and "ratelimit:" not in str(line)
    finished = [r.msg for r in caplog.records if isinstance(r.msg, dict) and r.msg["event"] == "request_finished"]
    assert finished[-1]["status_code"] == 429 and finished[-1]["outcome"] == "rate_limited"


# --- Real Redis (EXPIRE ... NX needs Redis >= 7) ---------------------------------------------------


@pytest.mark.redis
def test_real_redis_window_counts_and_ttl():
    url = os.environ.get("REDIS_URL")
    if not url:
        pytest.skip("set REDIS_URL to run against a real Redis")
    from redis.asyncio import Redis

    async def scenario():
        redis = Redis.from_url(url, decode_responses=True)
        session = f"test-{uuid.uuid4().hex}"
        limiter = RedisRateLimiter(url, per_session=2, per_ip=100, client=redis, clock=Clock())
        try:
            decisions = [await limiter.check(session, "192.0.2.1") for _ in range(3)]
            key = f"ratelimit:session:{session}:{(WINDOW_START + 5) // 60}"
            ttl = await redis.ttl(key)
            return [d.allowed for d in decisions], ttl
        finally:
            keys = [k async for k in redis.scan_iter(match="ratelimit:*")]
            mine = [k for k in keys if session in k or "192.0.2.1" in k]
            if mine:
                await redis.delete(*mine)
            await redis.aclose()

    allowed, ttl = asyncio.run(scenario())
    assert allowed == [True, True, False]
    assert 0 < ttl <= 60
