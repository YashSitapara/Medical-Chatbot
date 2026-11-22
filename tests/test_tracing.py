"""Langfuse tracing: trace tree, masking, trace_id in responses, feedback scores.

The real Langfuse SDK runs here with an in-memory OpenTelemetry exporter: nothing is sent
over the network, and the exported span attributes are exactly what Langfuse would get.
"""

import asyncio
import json
import subprocess
import sys
import time

import httpx
import pytest
from fastapi.testclient import TestClient
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from opentelemetry.sdk.trace.export import SpanExportResult
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from app.config import Settings
from app.main import create_app
from app.memory import InMemoryChatMemory
from app.rag import PROMPT_SHA, PROMPT_VERSION, RAGService
from app.rate_limit import InMemoryRateLimiter
from app.schemas import Message
from app.tracing import FEEDBACK_SCORE_NAME, LangfuseTracer, NoopTracer, build_tracer, mask_pii, mask_text
from tests.conftest import OFF_TOPIC, StubVectorStore
from tests.test_api import parse_sse
from tests.test_resilience import ScriptedLLM, StallingStreamLLM

PUBLIC_KEY = "pk-lf-test-public"
SECRET_KEY = "sk-lf-test-secret-0123456789"


def tracing_settings(**overrides) -> Settings:
    values = {
        "pinecone_api_key": "test-key",
        "llm_provider": "ollama",
        "tracing_enabled": True,
        "langfuse_public_key": PUBLIC_KEY,
        "langfuse_secret_key": SECRET_KEY,
        "langfuse_base_url": "http://127.0.0.1:9",  # never contacted for spans (in-memory exporter)
    }
    return Settings(**{**values, **overrides})


class KeepAliveTracer(LangfuseTracer):
    """The SDK keeps one client per public key; app shutdown must not close it between tests."""

    def shutdown(self) -> None:
        self.client.flush()


@pytest.fixture(scope="session")
def exporter():
    return InMemorySpanExporter()


@pytest.fixture(scope="session")
def tracer(exporter):
    built = build_tracer(tracing_settings(), span_exporter=exporter)
    return KeepAliveTracer(built.client, built.settings, built._metadata)


@pytest.fixture
def spans(tracer, exporter):
    """Call after the requests: returns {name: [span, ...]} for this test only."""
    tracer.client.flush()  # spans left buffered by an earlier test
    exporter.clear()

    def collect():
        tracer.client.flush()
        by_name: dict[str, list] = {}
        for span in exporter.get_finished_spans():
            by_name.setdefault(span.name, []).append(span)
        return by_name

    return collect


def make_app(tracer, llm=None, results=None, memory=None, limiter=None, **overrides):
    settings = tracing_settings(**overrides)
    rag = RAGService(settings, llm=llm or FakeListChatModel(responses=["Acne is a skin disease [1]."]),
                     vectorstore=StubVectorStore() if results is None else StubVectorStore(results))
    rag.retry_backoff_initial = rag.retry_backoff_max = 0
    app = create_app()
    app.state.rag = rag
    app.state.memory = memory or InMemoryChatMemory()
    app.state.rate_limiter = limiter
    app.state.tracer = tracer
    client = TestClient(app, raise_server_exceptions=False)
    client.__enter__()
    return client, app.state.memory


def attr(span, key, default=None):
    value = span.attributes.get(key, default)
    if isinstance(value, str):  # v5 JSON-encodes input / output / metadata values
        try:
            return json.loads(value)
        except ValueError:
            return value
    return value


def obs(span, field):
    return attr(span, f"langfuse.observation.{field}")


def hex_id(n: int, width: int) -> str:
    return format(n, f"0{width}x")


# --- Masking ------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text, expected",
    [
        ("mail me at priya.s+test@example.co.in please", "mail me at [EMAIL] please"),
        ("call +91 98765 43210", "call [PHONE]"),
        ("call +91-9876543210 now", "call [PHONE] now"),
        ("my number is 9876543210.", "my number is [PHONE]."),
        ("landline style 098765-43210", "landline style [PHONE]"),
        ("UK +44 20 7946 0958", "UK [PHONE]"),
        ("US +1 (415) 555-2671", "US [PHONE]"),
        ("aadhaar 2345 6789 0123", "aadhaar [AADHAAR]"),
        ("aadhaar 234567890123", "aadhaar [AADHAAR]"),
        ("aadhaar 2345-6789-0123", "aadhaar [AADHAAR]"),
        ("PAN ABCDE1234F", "PAN [PAN]"),
        ("pan abcde1234f", "pan [PAN]"),
    ],
)
def test_pii_is_masked(text, expected):
    assert mask_text(text) == expected


@pytest.mark.parametrize(
    "text",
    [
        "Take 500 mg twice a day",
        "See p. 43 of the book",
        "Published in 2023",
        "Score 0.734 and 0.91",
        "Acne is common [1][3].",
        "chunk c1746cfe590437d3",
        "Blood pressure 120/80, glucose 5.6 mmol/L",
        "Between 1990 and 2020, 1234 cases",
        "dose of 1000000 units",
    ],
)
def test_medical_text_is_not_masked(text):
    assert mask_text(text) == text


def test_mask_walks_containers_and_keeps_numbers():
    data = {"q": "email a@b.com", "chunks": [{"score": 0.9, "page": 12, "passed": True, "id": None}], "n": 3}
    assert mask_pii(data=data) == {
        "q": "email [EMAIL]",
        "chunks": [{"score": 0.9, "page": 12, "passed": True, "id": None}],
        "n": 3,
    }


# --- Trace structure ------------------------------------------------------------------------


def test_chat_trace_has_condense_retrieve_generate_under_one_root(tracer, spans):
    client, _ = make_app(tracer)
    r = client.post("/api/chat", json={"message": "What is acne?", "session_id": "sess-1"})
    assert r.status_code == 200
    trace_id = r.json()["trace_id"]
    s = spans()

    assert set(s) == {"chat", "condense", "retrieve", "generate"}
    root = s["chat"][0]
    assert root.parent is None and hex_id(root.context.trace_id, 32) == trace_id
    for name in ("condense", "retrieve", "generate"):
        child = s[name][0]
        assert child.context.trace_id == root.context.trace_id
        assert child.parent.span_id == root.context.span_id
        assert child.attributes["session.id"] == "sess-1"

    assert root.attributes["session.id"] == "sess-1"
    assert root.attributes["langfuse.trace.name"] == "chat"
    meta = {k.rsplit(".", 1)[1]: attr(root, k) for k in root.attributes if k.startswith("langfuse.trace.metadata.")}
    assert meta == {
        "request_id": r.headers["X-Request-ID"],
        "prompt_version": PROMPT_VERSION,
        "prompt_sha": PROMPT_SHA,
        "llm_provider": "ollama",
        "llm_model": "gemma3:4b",
        "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
        "index": "medical-chatbot",
        "k": 4,
        "min_score": 0.4,
    }
    assert obs(root, "input") == {"question": "What is acne?"}
    assert obs(root, "output") == {"answer": "Acne is a skin disease [1].", "cited": [1]}
    assert obs(root, "metadata.outcome") == "ok"
    assert obs(root, "metadata.condense_strategy") == "skipped"


def test_generate_records_full_prompt_model_and_output(tracer, spans):
    client, _ = make_app(tracer)
    client.post("/api/chat", json={"message": "What is acne?"})
    gen = spans()["generate"][0]
    assert obs(gen, "type") == "generation"
    assert gen.attributes["langfuse.observation.model.name"] == "gemma3:4b"
    assert obs(gen, "model.parameters") == {"temperature": 0.0}
    messages = obs(gen, "input")
    assert [m["role"] for m in messages] == ["system", "user"]
    assert "[1] (Medical_book.pdf, p. 12)\nAcne is a common skin disease." in messages[0]["content"]
    assert messages[1]["content"] == "What is acne?"
    assert obs(gen, "output") == "Acne is a skin disease [1]."


def test_generate_records_token_usage():
    from langchain_core.messages import AIMessage

    from app.rag import _usage

    assert _usage(AIMessage(content="x", usage_metadata={"input_tokens": 12, "output_tokens": 3,
                                                        "total_tokens": 15}).usage_metadata) == {"input": 12, "output": 3}
    assert _usage(None) is None


def test_retrieve_records_every_chunk_with_threshold_flag(tracer, spans):
    client, _ = make_app(tracer)
    client.post("/api/chat", json={"message": "What is acne?"})
    ret = spans()["retrieve"][0]
    assert obs(ret, "type") == "retriever"
    assert obs(ret, "input") == {"query": "What is acne?", "query_kind": "standalone"}
    chunks = obs(ret, "output")["chunks"]
    assert [(c["score"], c["page"], c["passed"]) for c in chunks] == [
        (0.91, 12, True), (0.85, 13, True), (0.4, None, True), (0.2, 99, False),
    ]
    assert all(len(c["id"]) == 16 and c["source"] == "Medical_book.pdf" for c in chunks)
    assert obs(ret, "metadata.attempts") == 1 and obs(ret, "metadata.k") == 4


def test_condense_records_history_and_rewrite(tracer, spans):
    memory = InMemoryChatMemory()
    asyncio.run(memory.append("s", "What is acne?", "Acne is a skin disease [1]."))
    llm = FakeListChatModel(responses=["Who usually gets acne?", "Teenagers [1]."])
    client, _ = make_app(tracer, llm=llm, memory=memory)
    client.post("/api/chat", json={"message": "Who gets it?", "session_id": "s"})
    s = spans()
    cond = s["condense"][0]
    assert obs(cond, "type") == "generation"
    assert obs(cond, "input") == {"transcript": "User: What is acne?\nAssistant: Acne is a skin disease.",
                                  "question": "Who gets it?"}
    out = obs(cond, "output")
    assert out["strategy"] == "llm" and out["standalone_question"] == "Who usually gets acne?"
    assert obs(s["retrieve"][0], "input")["query"] == "Who usually gets acne?"
    assert obs(s["chat"][0], "metadata.condense_strategy") == "llm"


def test_rejected_rewrite_shows_warning_and_both_searches(tracer, spans):
    memory = InMemoryChatMemory()
    asyncio.run(memory.append("s", "What is acne?", "Acne [1]."))
    # The "rewrite" is an answer with a citation: rejected -> dual retrieval.
    llm = FakeListChatModel(responses=["Teenagers get acne [1].", "Teenagers [1]."])
    client, _ = make_app(tracer, llm=llm, memory=memory)
    client.post("/api/chat", json={"message": "Who gets it?", "session_id": "s"})
    s = spans()
    cond = s["condense"][0]
    assert obs(cond, "level") == "WARNING"
    assert obs(cond, "output")["reason"] == "has_citation"
    assert obs(cond, "output")["raw_rewrite"] == "Teenagers get acne [1]."
    parent = s["retrieve"][0]
    assert obs(parent, "metadata.strategy") == "fallback_dual"
    kids = {sp.name: sp for sp in s["retrieve.plain"] + s["retrieve.combined"]}
    assert all(k.parent.span_id == parent.context.span_id for k in kids.values())
    assert obs(kids["retrieve.plain"], "input")["query"] == "Who gets it?"
    assert obs(kids["retrieve.combined"], "input")["query"] == "What is acne? Who gets it?"
    assert kids["retrieve.combined"].attributes["session.id"] == "s"


def test_condense_timeout_is_recorded(tracer, spans):
    memory = InMemoryChatMemory()
    asyncio.run(memory.append("s", "What is acne?", "Acne [1]."))
    llm = ScriptedLLM(responses=["x", "Teenagers [1]."], fail_calls=1, delay=1)
    client, _ = make_app(tracer, llm=llm, memory=memory, condense_timeout_seconds=0.05)
    assert client.post("/api/chat", json={"message": "Who gets it?", "session_id": "s"}).status_code == 200
    cond = spans()["condense"][0]
    assert obs(cond, "level") == "WARNING" and obs(cond, "output")["reason"] == "timeout"


def test_no_context_has_no_generation(tracer, spans):
    client, _ = make_app(tracer, results=OFF_TOPIC)
    client.post("/api/chat", json={"message": "How do I bake bread?"})
    s = spans()
    assert "generate" not in s
    assert obs(s["chat"][0], "metadata.outcome") == "no_context"
    assert obs(s["chat"][0], "level") is None


def test_dependency_error_marks_root_without_exception_text(tracer, spans):
    llm = ScriptedLLM(responses=["x"], fail_calls=1, error=httpx.ConnectError("secret host 10.0.0.5 refused"))
    client, _ = make_app(tracer, llm=llm)
    r = client.post("/api/chat", json={"message": "What is acne?"})
    assert r.status_code == 503
    s = spans()
    root, gen = s["chat"][0], s["generate"][0]
    assert obs(root, "level") == "ERROR" and obs(root, "status_message") == "LLMUnavailableError"
    assert obs(root, "metadata.outcome") == "dependency_unavailable"
    assert obs(gen, "level") == "ERROR"
    everything = json.dumps([dict(sp.attributes) for sp in exporter_spans(s)], default=str)
    assert "10.0.0.5" not in everything and "refused" not in everything


def exporter_spans(by_name):
    return [sp for group in by_name.values() for sp in group]


def test_pii_never_leaves_the_app(tracer, spans):
    memory = InMemoryChatMemory()
    asyncio.run(memory.append("s", "I am priya@example.com, 9876543210", "Hello [1]."))
    llm = FakeListChatModel(responses=["What is acne for PAN ABCDE1234F?", "Acne [1]."])
    client, _ = make_app(tracer, llm=llm, memory=memory)
    client.post("/api/chat", json={"message": "What is acne? Aadhaar 2345 6789 0123", "session_id": "s"})
    everything = json.dumps([dict(sp.attributes) for sp in exporter_spans(spans())], default=str)
    for raw in ("priya@example.com", "9876543210", "ABCDE1234F", "2345 6789 0123"):
        assert raw not in everything
    for token in ("[EMAIL]", "[PHONE]", "[PAN]", "[AADHAAR]"):
        assert token in everything


# --- Streaming --------------------------------------------------------------------------------


def test_stream_done_has_trace_id_and_same_tree(tracer, spans):
    client, _ = make_app(tracer)
    events = parse_sse(client.post("/api/chat/stream", json={"message": "What is acne?"}).text)
    done = dict(events)["done"]
    s = spans()
    assert set(s) == {"chat-stream", "condense", "retrieve", "generate"}
    root = s["chat-stream"][0]
    assert done["trace_id"] == hex_id(root.context.trace_id, 32)
    gen = s["generate"][0]
    assert gen.parent.span_id == root.context.span_id
    assert "langfuse.observation.completion_start_time" in gen.attributes
    assert obs(gen, "output") == "Acne is a skin disease [1]."
    assert obs(root, "metadata.outcome") == "ok"


def test_stream_timeout_marks_trace_and_cancels_generation(tracer, spans):
    class SlowStream(FakeListChatModel):
        async def astream(self, input, config=None, **kwargs):
            async for chunk in super().astream(input, config, **kwargs):
                yield chunk
                await asyncio.sleep(0.2)

    client, _ = make_app(tracer, llm=SlowStream(responses=["Acne is a skin disease [1]."]),
                         request_timeout_seconds=0.3)
    events = parse_sse(client.post("/api/chat/stream", json={"message": "What is acne?"}).text)
    error = events[-1][1]
    s = spans()
    root = s["chat-stream"][0]
    assert events[-1][0] == "error" and error["trace_id"] == hex_id(root.context.trace_id, 32)
    assert obs(root, "level") == "ERROR" and obs(root, "metadata.outcome") == "timeout"
    gen = s["generate"][0]
    assert obs(gen, "level") == "WARNING" and obs(gen, "status_message") == "cancelled"
    assert obs(gen, "output")  # the part generated before the cap is kept


def test_stream_provider_error_marks_generation(tracer, spans):
    client, _ = make_app(tracer, llm=StallingStreamLLM(responses=["Acne is a skin disease [1]."]))
    events = parse_sse(client.post("/api/chat/stream", json={"message": "What is acne?"}).text)
    assert events[-1][1]["code"] == "dependency_unavailable"
    s = spans()
    assert obs(s["generate"][0], "level") == "ERROR"
    assert obs(s["chat-stream"][0], "metadata.outcome") == "dependency_unavailable"


# --- History, disabled, Langfuse down -------------------------------------------------------------


def test_history_keeps_trace_id_and_old_messages_still_load(tracer, spans):
    client, memory = make_app(tracer)
    r = client.post("/api/chat", json={"message": "What is acne?", "session_id": "h"})
    messages = client.get("/api/sessions/h/history").json()["messages"]
    assert messages[0]["trace_id"] is None and messages[1]["trace_id"] == r.json()["trace_id"]
    old = '{"role": "assistant", "content": "hi", "created_at": "2026-01-01T00:00:00Z"}'
    assert Message.model_validate_json(old).trace_id is None


def test_tracing_disabled_returns_null_trace_and_no_feedback(spans):
    client, _ = make_app(NoopTracer())
    assert client.post("/api/chat", json={"message": "What is acne?"}).json()["trace_id"] is None
    done = dict(parse_sse(client.post("/api/chat/stream", json={"message": "What is acne?"}).text))["done"]
    assert done["trace_id"] is None
    r = client.post("/api/feedback", json={"trace_id": "a" * 32, "rating": 1})
    assert r.status_code == 503 and r.json()["code"] == "tracing_disabled"
    assert spans() == {}


def test_build_tracer_off_by_default_and_keys_required():
    assert isinstance(build_tracer(Settings(pinecone_api_key="k")), NoopTracer)
    with pytest.raises(ValueError, match="LANGFUSE_PUBLIC_KEY"):
        Settings(pinecone_api_key="k", tracing_enabled=True)


def test_unsampled_request_gets_no_trace_id():
    """LANGFUSE_SAMPLE_RATE=0: nothing is sent, so there is no trace to rate.

    Runs in a fresh process: OpenTelemetry has ONE global tracer provider (with the sampler)
    per process, created by the first Langfuse client, i.e. this module's session tracer.
    """
    code = """
import tests.conftest
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from tests.test_tracing import KeepAliveTracer, build_tracer, make_app, tracing_settings
exporter = InMemorySpanExporter()
built = build_tracer(tracing_settings(langfuse_sample_rate=0.0), span_exporter=exporter)
client, _ = make_app(KeepAliveTracer(built.client, built.settings, built._metadata))
r = client.post("/api/chat", json={"message": "What is acne?"})
built.client.flush()
assert r.status_code == 200, r.status_code
assert r.json()["trace_id"] is None, r.json()
assert len(exporter.get_finished_spans()) == 0
print("OK")
"""
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=60)
    assert result.stdout.strip().endswith("OK"), result.stderr[-2000:]


def test_langfuse_down_does_not_slow_or_fail_requests():
    class BrokenExporter(InMemorySpanExporter):
        def export(self, spans):
            time.sleep(0.5)  # a hung Langfuse
            raise ConnectionError("langfuse unreachable")

    exporter = BrokenExporter()
    broken = build_tracer(tracing_settings(langfuse_public_key="pk-lf-broken"), span_exporter=exporter)
    tracer = KeepAliveTracer(broken.client, broken.settings, broken._metadata)
    client, _ = make_app(tracer)
    start = time.perf_counter()
    for _ in range(3):
        assert client.post("/api/chat", json={"message": "What is acne?"}).status_code == 200
    assert time.perf_counter() - start < 0.5  # export runs in the background
    exporter.export = lambda spans: SpanExportResult.SUCCESS  # recover, so exit-time flush is quiet
    tracer.client.flush()


# --- Feedback ---------------------------------------------------------------------------------------


@pytest.fixture
def scores(tracer, monkeypatch):
    calls = []
    monkeypatch.setattr(tracer.client, "create_score", lambda **kw: calls.append(kw))
    return calls


@pytest.mark.parametrize("rating", [1, -1])
def test_feedback_becomes_an_idempotent_score(tracer, scores, rating):
    client, _ = make_app(tracer)
    trace_id = client.post("/api/chat", json={"message": "What is acne?"}).json()["trace_id"]
    r = client.post("/api/feedback", json={"trace_id": trace_id, "rating": rating,
                                           "comment": "  wrong, call me on 9876543210  "})
    assert r.status_code == 202 and r.json() == {"status": "accepted"}
    assert scores == [{
        "name": FEEDBACK_SCORE_NAME,
        "value": float(rating),
        "data_type": "NUMERIC",
        "trace_id": trace_id,
        "score_id": f"{trace_id}-{FEEDBACK_SCORE_NAME}",
        "comment": "wrong, call me on [PHONE]",
    }]


def test_feedback_without_comment(tracer, scores):
    client, _ = make_app(tracer)
    assert client.post("/api/feedback", json={"trace_id": "b" * 32, "rating": 1, "comment": "   "}).status_code == 202
    assert scores[0]["comment"] is None


@pytest.mark.parametrize(
    "body",
    [
        {"trace_id": "not-a-trace", "rating": 1},
        {"trace_id": "A" * 32, "rating": 1},
        {"trace_id": "a" * 32, "rating": 0},
        {"trace_id": "a" * 32, "rating": 2},
        {"trace_id": "a" * 32, "rating": 1, "comment": "x" * 501},
        {"rating": 1},
    ],
)
def test_feedback_validation(tracer, scores, body):
    client, _ = make_app(tracer)
    r = client.post("/api/feedback", json=body)
    assert r.status_code == 422 and r.json()["code"] == "validation_error"
    assert scores == []


def test_feedback_has_its_own_rate_limit(tracer, scores):
    limiter = InMemoryRateLimiter(per_session=10, per_ip=2)
    client, _ = make_app(tracer, limiter=limiter, feedback_rate_limit_per_ip_per_minute=3)
    codes = [client.post("/api/feedback", json={"trace_id": "c" * 32, "rating": 1}).status_code for _ in range(4)]
    assert codes == [202, 202, 202, 429]
    # The chat allowance (2 per IP here) is untouched by the feedback calls.
    assert [client.post("/api/chat", json={"message": "What is acne?"}).status_code for _ in range(2)] == [200, 200]


def test_feedback_failure_is_503(tracer, monkeypatch):
    def boom(**_):
        raise RuntimeError("queue full")

    monkeypatch.setattr(tracer.client, "create_score", boom)
    client, _ = make_app(tracer)
    r = client.post("/api/feedback", json={"trace_id": "d" * 32, "rating": -1})
    assert r.status_code == 503 and r.json()["code"] == "feedback_unavailable"


# --- Logs ---------------------------------------------------------------------------------------------


def test_logs_carry_trace_id(tracer, caplog):
    client, _ = make_app(tracer)
    trace_id = client.post("/api/chat", json={"message": "What is acne?"}).json()["trace_id"]
    finished = [r.msg for r in caplog.records if isinstance(r.msg, dict) and r.msg.get("event") == "request_finished"]
    retrieval = [r.msg for r in caplog.records if isinstance(r.msg, dict) and r.msg.get("event") == "retrieval_done"]
    assert finished and retrieval
    rendered = "\n".join(caplog.text.splitlines())
    assert trace_id in rendered


def test_langfuse_keys_are_redacted_from_logs():
    from app.log_config import Redactor

    redact = Redactor(log_content=False, secret_values=[SECRET_KEY])
    out = redact(None, None, {"event": "x", "detail": f"auth failed for {PUBLIC_KEY} / {SECRET_KEY}"})
    assert SECRET_KEY not in out["detail"] and PUBLIC_KEY not in out["detail"]
