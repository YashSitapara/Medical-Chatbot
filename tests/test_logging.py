"""Structured logging: request IDs, the per-request events, and what must never be logged."""

import io
import json
import logging

import fakeredis
import pytest
import structlog
from fastapi.testclient import TestClient
from langchain_core.language_models.fake_chat_models import FakeListChatModel, GenericFakeChatModel
from langchain_core.messages import AIMessage

from app.config import Settings
from app.log_config import configure_logging
from app.main import create_app
from app.memory import InMemoryChatMemory, RedisChatMemory
from app.rag import RAGService
from tests.conftest import OFF_TOPIC, StubVectorStore
from tests.test_api import parse_sse

QUESTION = "Jane Doe 1984 HIV status?"
REWRITE = "What is the HIV status of Jane Doe 1984?"
FAKE_PINECONE_KEY = "pcsk_TESTKEY_abcdefghijklmnop1234"


class ExplodingLLM(FakeListChatModel):
    """Raises an error whose message echoes the prompt - like a provider error might."""

    async def ainvoke(self, input, config=None, **kwargs):
        raise ValueError(f"provider rejected prompt containing: {input}")


@pytest.fixture
def make_logged_client():
    clients = []

    def _make(responses=None, *, log_content=False, llm=None, results=None, memory=None, log_level="INFO"):
        settings = Settings(
            pinecone_api_key=FAKE_PINECONE_KEY, llm_provider="ollama", log_content=log_content, log_level=log_level,
        )
        store = StubVectorStore() if results is None else StubVectorStore(results)
        rag = RAGService(settings, llm=llm or FakeListChatModel(responses=responses or []), vectorstore=store)
        app = create_app()
        buf = io.StringIO()
        configure_logging(settings, stream=buf)  # after create_app, which configures stdout
        app.state.rag = rag
        app.state.memory = memory or InMemoryChatMemory()
        client = TestClient(app, raise_server_exceptions=False)
        client.__enter__()
        clients.append(client)

        def lines() -> list[dict]:
            return [json.loads(line) for line in buf.getvalue().splitlines() if line.strip()]

        return client, lines, buf

    yield _make
    for c in clients:
        c.__exit__(None, None, None)
    configure_logging(Settings(pinecone_api_key="test-key"))  # back to stdout for other tests


def events(lines, request_id=None):
    return [ln for ln in lines if request_id is None or ln.get("request_id") == request_id]


def names(lines):
    return [ln["event"] for ln in lines if ln["event"] in REQUEST_EVENTS]


REQUEST_EVENTS = ("request_started", "query_condensed", "retrieval_done", "llm_done", "request_finished")


# --- Request IDs ----------------------------------------------------------------------


def test_request_id_generated_and_returned(make_logged_client):
    client, lines, _ = make_logged_client()
    r = client.get("/health")
    rid = r.headers["X-Request-ID"]
    assert len(rid) == 32 and int(rid, 16) >= 0


def test_valid_incoming_request_id_is_reused(make_logged_client):
    client, lines, _ = make_logged_client(["Acne is a skin disease [1]."])
    r = client.post("/api/chat", json={"message": "What is acne?"}, headers={"X-Request-ID": "test-123"})
    assert r.headers["X-Request-ID"] == "test-123"
    mine = events(lines(), "test-123")
    assert names(mine) == ["request_started", "query_condensed", "retrieval_done", "llm_done", "request_finished"]


# HTTP clients refuse to send raw newlines; JSON/quote characters are the realistic forging attempt.
@pytest.mark.parametrize("bad", ["two words", "x" * 200, '{"event":"forged"}', "ünïcode".encode()])
def test_invalid_incoming_request_id_is_replaced(make_logged_client, bad):
    client, lines, _ = make_logged_client()
    r = client.get("/health", headers={"X-Request-ID": bad})
    rid = r.headers["X-Request-ID"]
    assert len(rid) == 32 and int(rid, 16) >= 0


def test_request_id_on_sse_503_and_404(make_logged_client):
    client, _, _ = make_logged_client(["Acne [1]."])
    assert client.post("/api/chat/stream", json={"message": "What is acne?"}).headers.get("X-Request-ID")
    assert client.get("/api/sessions/nope/history").headers.get("X-Request-ID")

    server = fakeredis.FakeServer()
    redis_mem = RedisChatMemory(client=fakeredis.FakeAsyncRedis(server=server, decode_responses=True))
    client2, lines, _ = make_logged_client(memory=redis_mem)
    server.connected = False  # after startup (the app refuses to start without Redis)
    r = client2.post("/api/chat", json={"message": "What is acne?"})
    assert r.status_code == 503 and r.headers.get("X-Request-ID")
    finished = [ln for ln in lines() if ln["event"] == "request_finished"][-1]
    assert finished["status_code"] == 503 and finished["outcome"] == "dependency_unavailable"
    assert finished["level"] == "warning"


def test_unhandled_exception_gets_500_with_request_id(make_logged_client):
    client, lines, _ = make_logged_client()
    client.app.state.memory = None  # breaks /api/sessions/... with an AttributeError
    r = client.delete("/api/sessions/s1", headers={"X-Request-ID": "boom-1"})
    assert r.status_code == 500
    assert r.headers["X-Request-ID"] == "boom-1" and r.json()["request_id"] == "boom-1"
    finished = [ln for ln in events(lines(), "boom-1") if ln["event"] == "request_finished"]
    assert finished[0]["outcome"] == "exception" and "AttributeError" in finished[0]["exception"]


# --- Event content ----------------------------------------------------------------------


def test_followup_emits_all_events_with_shared_ids(make_logged_client):
    client, lines, _ = make_logged_client(["Acne is a skin disease [1].", "What causes acne?", "Clogged pores [2]."])
    sid = client.post("/api/chat", json={"message": "What is acne?"}).json()["session_id"]
    r = client.post("/api/chat", json={"message": "What causes it?", "session_id": sid})
    rid = r.headers["X-Request-ID"]

    mine = events(lines(), rid)
    assert names(mine) == list(REQUEST_EVENTS)
    assert all(ln["session_id"] == sid and isinstance(ln["pid"], int) and ln["timestamp"] for ln in mine)
    by = {ln["event"]: ln for ln in mine}

    assert by["request_started"].items() >= {"route": "/api/chat", "method": "POST", "stream": False,
                                              "new_session": False, "question_chars": len("What causes it?")}.items()
    assert by["query_condensed"]["changed"] is True and by["query_condensed"]["condense_strategy"] == "llm"
    assert by["query_condensed"]["history_messages"] == 2 and by["query_condensed"]["latency_ms"] >= 0
    rd = by["retrieval_done"]
    assert rd["chunks_retrieved"] == 4 and rd["chunks_passed"] == 3 and rd["min_score"] == 0.4
    assert rd["top_score"] == 0.91 and rd["scores"] == [0.91, 0.85, 0.4, 0.2]
    assert by["llm_done"]["provider"] == "ollama" and by["llm_done"]["model"] == "gemma3:4b"
    assert by["llm_done"]["input_tokens"] is None  # FakeListChatModel reports no usage
    fin = by["request_finished"]
    assert fin["status_code"] == 200 and fin["outcome"] == "ok" and fin["route"] == "/api/chat"
    assert fin["latency_ms"] >= by["llm_done"]["latency_ms"]


def test_first_turn_condense_is_skipped(make_logged_client):
    client, lines, _ = make_logged_client(["Acne [1]."])
    client.post("/api/chat", json={"message": "What is acne?"})
    cond = next(ln for ln in lines() if ln["event"] == "query_condensed")
    assert cond["condense_strategy"] == "skipped" and cond["changed"] is False and cond["history_messages"] == 0


def test_token_usage_is_logged(make_logged_client):
    usage = {"input_tokens": 120, "output_tokens": 15, "total_tokens": 135}
    llm = GenericFakeChatModel(messages=iter([AIMessage(content="Acne [1].", usage_metadata=usage)]))
    client, lines, _ = make_logged_client(llm=llm)
    client.post("/api/chat", json={"message": "What is acne?"})
    llm_done = next(ln for ln in lines() if ln["event"] == "llm_done")
    assert llm_done["input_tokens"] == 120 and llm_done["output_tokens"] == 15


def test_stream_finished_after_whole_stream(make_logged_client):
    llm = FakeListChatModel(responses=["Acne is a skin disease [1]."], sleep=0.01)
    client, lines, _ = make_logged_client(llm=llm)
    r = client.post("/api/chat/stream", json={"message": "What is acne?"})
    assert parse_sse(r.text)[-1][0] == "done"
    mine = events(lines(), r.headers["X-Request-ID"])
    assert names(mine) == ["request_started", "query_condensed", "retrieval_done", "llm_done", "request_finished"]
    by = {ln["event"]: ln for ln in mine}
    assert by["request_started"]["stream"] is True
    assert by["llm_done"]["ttft_ms"] <= by["llm_done"]["latency_ms"]
    # Measured at the LAST body chunk, so it covers token streaming (~0.27 s here).
    assert by["request_finished"]["latency_ms"] >= by["llm_done"]["latency_ms"] >= 200
    assert by["request_finished"]["session_id"]  # set inside the stream's child task, still visible


def test_stream_failure_is_200_with_error_outcome(make_logged_client):
    llm = FakeListChatModel(responses=["Acne is a skin disease [1]."], error_on_chunk_number=3)
    client, lines, _ = make_logged_client(llm=llm)
    r = client.post("/api/chat/stream", json={"message": "What is acne?"})
    last = parse_sse(r.text)[-1]
    assert last[0] == "error" and last[1]["request_id"] == r.headers["X-Request-ID"]
    fin = [ln for ln in lines() if ln["event"] == "request_finished"][-1]
    assert fin["status_code"] == 200 and fin["outcome"] == "error"


def test_out_of_scope_has_no_llm_done(make_logged_client):
    client, lines, _ = make_logged_client([], results=OFF_TOPIC)
    client.post("/api/chat", json={"message": "Who won the cup?"})
    assert "llm_done" not in names(lines())
    fin = next(ln for ln in lines() if ln["event"] == "request_finished")
    assert fin["outcome"] == "no_context" and fin["status_code"] == 200


def test_route_is_template_not_raw_path(make_logged_client):
    client, lines, _ = make_logged_client()
    client.get("/api/sessions/secret-session-xyz/history")
    fin = next(ln for ln in lines() if ln["event"] == "request_finished")
    assert fin["route"] == "/api/sessions/{session_id}/history" and fin["status_code"] == 404
    # A client error is not a server failure: INFO, not WARNING.
    assert fin["outcome"] == "client_error" and fin["level"] == "info"


def test_probes_quiet_when_healthy_loud_when_not(make_logged_client):
    client, lines, _ = make_logged_client()
    client.get("/health")
    assert not [ln for ln in lines() if ln["event"] == "request_finished"]

    server = fakeredis.FakeServer()
    server.connected = False
    client.app.state.memory = RedisChatMemory(client=fakeredis.FakeAsyncRedis(server=server, decode_responses=True))
    client.get("/health")
    fin = [ln for ln in lines() if ln["event"] == "request_finished"]
    assert fin and fin[0]["level"] == "warning" and fin[0]["status_code"] == 503


# --- Privacy ----------------------------------------------------------------------------


def _chat_twice(client, stream: bool):
    path = "/api/chat/stream" if stream else "/api/chat"
    r = client.post(path, json={"message": "Tell me about HIV."})
    sid = parse_sse(r.text)[0][1]["session_id"] if stream else r.json()["session_id"]
    client.post(path, json={"message": QUESTION, "session_id": sid})


@pytest.mark.parametrize("stream", [False, True])
def test_questions_never_logged_by_default(make_logged_client, stream):
    client, _, buf = make_logged_client(["HIV is a virus [1].", REWRITE, "Unknown [1]."])
    _chat_twice(client, stream)
    out = buf.getvalue()
    assert "request_started" in out and "query_condensed" in out
    for text in (QUESTION, REWRITE, "Tell me about HIV", "Jane Doe"):
        assert text not in out


def test_rejected_rewrite_text_never_logged(make_logged_client):
    answer_like = "Jane Doe 1984 is HIV positive according to the record [1]."
    client, lines, buf = make_logged_client(["HIV is a virus [1].", answer_like, "Unknown [1]."])
    _chat_twice(client, stream=False)
    rejected = next(ln for ln in lines() if ln["event"] == "rewrite_rejected")
    assert rejected["reason"] == "has_citation" and "rewrite_raw" not in rejected
    cond = [ln for ln in lines() if ln["event"] == "query_condensed"][-1]
    assert cond["condense_strategy"] == "fallback_dual" and cond["reason"] == "has_citation"
    assert "Jane Doe" not in buf.getvalue()


def test_question_scrubbed_from_exception_messages(make_logged_client):
    client, lines, buf = make_logged_client(llm=ExplodingLLM(responses=[]))
    r = client.post("/api/chat", json={"message": QUESTION})
    assert r.status_code == 500  # a non-transient error is a bug, not an outage
    failed = next(ln for ln in lines() if ln["event"] == "request_finished")
    assert "ValueError" in failed["exception"] and "[question]" in failed["exception"]
    assert "Jane Doe" not in buf.getvalue()


def test_log_content_on_logs_questions_only_in_allowed_fields(make_logged_client):
    client, lines, buf = make_logged_client(["HIV is a virus [1].", REWRITE, "Unknown [1]."], log_content=True)
    _chat_twice(client, stream=False)
    started = [ln for ln in lines() if ln["event"] == "request_started"][-1]
    cond = [ln for ln in lines() if ln["event"] == "query_condensed"][-1]
    assert started["question"] == QUESTION and cond["condensed_question"] == REWRITE
    holders = [ln["event"] for ln in lines() if QUESTION in json.dumps(ln) or REWRITE in json.dumps(ln)]
    assert sorted(holders) == ["query_condensed", "request_started"]
    assert "Unknown [1]" not in buf.getvalue()  # answers are never logged


# --- Secrets ----------------------------------------------------------------------------


def test_secrets_redacted_everywhere(make_logged_client):
    client, _, buf = make_logged_client()
    log = structlog.get_logger("test")
    log.info("leaky", api_key="abc123", detail=f"key={FAKE_PINECONE_KEY} url=redis://:hunter2@redis:6379/0",
             header="Authorization: Bearer eyJhbGciOi.payload.sig", openai="sk-proj-ABCDEFGHIJKLMNOPQRSTUV")
    logging.getLogger("thirdparty").warning("connecting with api_key=%s to %s", "plainsecretvalue",
                                            "https://user:pw123@host")
    out = buf.getvalue()
    for secret in ("abc123", FAKE_PINECONE_KEY, "TESTKEY", "hunter2", "eyJhbGciOi", "sk-proj-ABCDEF",
                   "plainsecretvalue", "pw123"):
        assert secret not in out
    assert out.count("[REDACTED]") >= 5


def test_configured_key_redacted_even_without_known_prefix():
    buf = io.StringIO()
    configure_logging(Settings(pinecone_api_key="customKEYwithoutprefix9", llm_provider="ollama"), stream=buf)
    try:
        structlog.get_logger("t").info("x", detail="using customKEYwithoutprefix9 now")
        assert "customKEYwithoutprefix9" not in buf.getvalue()
    finally:
        configure_logging(Settings(pinecone_api_key="test-key"))


def test_token_counts_are_not_mistaken_for_secrets(make_logged_client):
    client, _, buf = make_logged_client()
    structlog.get_logger("t").info("x", input_tokens=5, output_tokens=7)
    line = json.loads(buf.getvalue().splitlines()[-1])
    assert line["input_tokens"] == 5 and line["output_tokens"] == 7
