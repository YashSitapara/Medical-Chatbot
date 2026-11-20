"""Timeouts, retries, fallbacks and error mapping (503 dependency / 504 cap / 500 bug)."""

import asyncio
import time

import httpx
import ollama
import openai
import pytest
from fastapi.testclient import TestClient
from langchain_core.documents import Document
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from pinecone.exceptions import PineconeApiException, ServiceException, UnauthorizedException
from urllib3.exceptions import ProtocolError

from app.config import Settings
from app.main import create_app
from app.memory import InMemoryChatMemory
from app.rag import RAGService, build_llm
from app.schemas import Message
from tests.conftest import DOCS, StubVectorStore
from tests.test_api import parse_sse


class ScriptedLLM(FakeListChatModel):
    """FakeListChatModel whose first `fail_calls` ainvoke calls raise or hang.

    The condense chain calls the LLM first (when there is history), so failing the first
    call fails the rewrite while the answer still works.
    """

    fail_calls: int = 0
    error: BaseException | None = None
    delay: float = 0.0
    calls: int = 0

    async def ainvoke(self, input, config=None, **kwargs):
        self.calls += 1
        if self.calls <= self.fail_calls:
            if self.delay:
                await asyncio.sleep(self.delay)
            if self.error is not None:
                raise self.error
        return await super().ainvoke(input, config, **kwargs)


def make(llm=None, results=None, memory=None, **overrides):
    settings = Settings(pinecone_api_key="test-key", llm_provider="ollama", **overrides)
    store = StubVectorStore() if results is None else StubVectorStore(results)
    rag = RAGService(settings, llm=llm or FakeListChatModel(responses=["Acne [1]."]), vectorstore=store)
    rag.retry_backoff_initial = rag.retry_backoff_max = 0
    app = create_app()
    app.state.rag = rag
    app.state.memory = memory or InMemoryChatMemory()
    app.state.rate_limiter = None
    client = TestClient(app, raise_server_exceptions=False)
    return client, rag, store, app.state.memory


def events_named(caplog, name):
    return [r.msg for r in caplog.records if isinstance(r.msg, dict) and r.msg.get("event") == name]


def history(*pairs):
    msgs = []
    for q, a in pairs:
        msgs += [Message(role="user", content=q), Message(role="assistant", content=a)]
    return msgs


def doc(text, page):
    return Document(page_content=text, metadata={"source": "Medical_book.pdf", "page": float(page)})


# --- LLM settings -----------------------------------------------------------------------


def test_openai_gets_timeout_and_retries_from_settings():
    llm = build_llm(Settings(pinecone_api_key="k", llm_provider="openai", openai_api_key="sk-test",
                             llm_timeout_seconds=30, llm_max_retries=2))
    assert llm.request_timeout == 30 and llm.max_retries == 2


def test_ollama_gets_timeout_from_settings():
    llm = build_llm(Settings(pinecone_api_key="k", llm_provider="ollama", llm_timeout_seconds=12))
    assert llm.client_kwargs == {"timeout": 12}


# --- Pinecone retries -----------------------------------------------------------------------


def test_retrieval_retries_5xx_then_succeeds(caplog):
    client, _, store, _ = make()
    store.index.query_errors = [ServiceException(status=503, reason="x"), ProtocolError("reset")]
    r = client.post("/api/chat", json={"message": "What is acne?"})
    assert r.status_code == 200
    assert len(store.index.query_calls) == 3
    assert store.index.query_calls[0]["_request_timeout"] == 5  # the HTTP call itself is limited
    assert [e["error"] for e in events_named(caplog, "retrieval_retry")] == ["ServiceException", "ProtocolError"]


def test_retrieval_timeouts_exhaust_retries_then_503():
    client, _, store, _ = make(retrieval_timeout_seconds=0.05)
    store.index.query_delay = 0.3
    start = time.perf_counter()
    r = client.post("/api/chat", json={"message": "What is acne?"})
    assert r.status_code == 503 and time.perf_counter() - start < 1.0
    assert r.json()["code"] == "dependency_unavailable" and r.json()["request_id"] == r.headers["X-Request-ID"]
    assert len(store.index.query_calls) == 3


@pytest.mark.parametrize("error", [UnauthorizedException(status=401, reason="bad key"),
                                   PineconeApiException(status=404, reason="no index")])
def test_retrieval_4xx_is_not_retried_and_is_a_500(error):
    client, _, store, _ = make()
    store.index.query_errors = [error, error, error]
    r = client.post("/api/chat", json={"message": "What is acne?"})
    assert len(store.index.query_calls) == 1  # retrying a wrong request can't help
    assert r.status_code == 500 and r.json()["code"] == "internal_error" and r.json()["request_id"]


# --- Condense fallback: dual retrieval -------------------------------------------------------


ACNE = [(doc("Acne affects mostly teenagers.", 10), 0.82), (doc("Acne is a skin disease.", 11), 0.78)]
VAGUE = [(doc("Some people get it from relatives.", 50), 0.45)]
MIGRAINE = [(doc("Migraines are caused by brain chemistry.", 30), 0.81), (doc("Migraine triggers.", 31), 0.70)]
DIABETES = [(doc("Diabetes affects blood sugar.", 20), 0.66), (doc("Insulin treats diabetes.", 21), 0.62)]


def test_pronoun_followup_with_failed_rewrite_uses_combined_search():
    llm = ScriptedLLM(responses=["Teenagers [1]."], fail_calls=1, error=ValueError("model crashed"))
    results = {"Who usually gets it?": VAGUE, "What is acne? Who usually gets it?": ACNE}
    client, _, store, memory = make(llm=llm, results=results)
    asyncio.run(memory.append("s", "What is acne?", "Acne is a skin disease [1]."))
    r = client.post("/api/chat", json={"message": "Who usually gets it?", "session_id": "s"})
    assert r.status_code == 200
    snippets = [c["snippet"] for c in r.json()["citations"]]
    assert snippets[:2] == ["Acne affects mostly teenagers.", "Acne is a skin disease."]


def test_topic_switch_with_failed_rewrite_prefers_plain_search():
    llm = ScriptedLLM(responses=["Brain chemistry [1]."], fail_calls=1, error=ValueError("model crashed"))
    results = {
        "What causes migraines?": MIGRAINE,
        "What is diabetes? What causes migraines?": DIABETES + [(MIGRAINE[1][0], 0.55)],
    }
    client, _, _, memory = make(llm=llm, results=results)
    asyncio.run(memory.append("s", "What is diabetes?", "Diabetes affects blood sugar [1]."))
    citations = client.post("/api/chat", json={"message": "What causes migraines?", "session_id": "s"}).json()["citations"]
    pages = [c["page"] for c in citations]
    assert pages[:2] == [30, 31]  # both migraine chunks first; no diabetes chunk outranks them
    assert [c["score"] for c in citations if c["page"] == 31] == [0.7]  # duplicate kept once, best score


@pytest.mark.parametrize("failure", ["raises", "times_out"])
def test_failed_or_timed_out_rewrite_continues_with_dual_retrieval(caplog, failure):
    if failure == "raises":
        llm = ScriptedLLM(responses=["Teens [1]."], fail_calls=1, error=httpx.ConnectError("refused"))
    else:
        llm = ScriptedLLM(responses=["Teens [1]."], fail_calls=1, delay=1.0)
    client, _, store, memory = make(llm=llm, condense_timeout_seconds=0.05)
    asyncio.run(memory.append("s", "What is acne?", "Acne [1]."))
    r = client.post("/api/chat", json={"message": "Who gets it?", "session_id": "s"})
    assert r.status_code == 200 and r.json()["answer"] == "Teens [1]."
    cond = events_named(caplog, "query_condensed")[-1]
    assert cond["condense_strategy"] == "fallback_dual"
    assert cond["reason"] == ("call_failed" if failure == "raises" else "timeout")
    assert sorted(e["query_kind"] for e in events_named(caplog, "retrieval_done")) == ["combined", "plain"]


def test_one_fallback_search_failing_still_answers():
    llm = ScriptedLLM(responses=["Teens [1]."], fail_calls=1, error=ValueError("x"))
    results = {"Who gets it?": ServiceException(status=503, reason="down"), "What is acne? Who gets it?": ACNE}
    client, _, _, memory = make(llm=llm, results=results)
    asyncio.run(memory.append("s", "What is acne?", "Acne [1]."))
    r = client.post("/api/chat", json={"message": "Who gets it?", "session_id": "s"})
    assert r.status_code == 200 and r.json()["citations"][0]["page"] == 10


def test_both_fallback_searches_failing_is_503():
    llm = ScriptedLLM(responses=["x"], fail_calls=1, error=ValueError("x"))
    down = ServiceException(status=503, reason="down")
    client, _, _, memory = make(llm=llm, results={"Who gets it?": down, "What is acne? Who gets it?": down})
    asyncio.run(memory.append("s", "What is acne?", "Acne [1]."))
    assert client.post("/api/chat", json={"message": "Who gets it?", "session_id": "s"}).status_code == 503


def test_previous_question_is_capped_at_300_chars():
    llm = ScriptedLLM(responses=["x"], fail_calls=1, error=ValueError("x"))
    client, _, store, memory = make(llm=llm)
    asyncio.run(memory.append("s", "A" * 1000, "Answer [1]."))
    client.post("/api/chat", json={"message": "Who gets it?", "session_id": "s"})
    assert sorted(store.queries, key=len) == ["Who gets it?", "A" * 300 + " Who gets it?"]


# --- Overall request cap -----------------------------------------------------------------


def test_chat_over_the_cap_is_504():
    llm = ScriptedLLM(responses=["late"], fail_calls=1, delay=2.0)  # first call = the answer (no history)
    client, _, _, _ = make(llm=llm, request_timeout_seconds=0.2)
    start = time.perf_counter()
    r = client.post("/api/chat", json={"message": "What is acne?"})
    assert r.status_code == 504 and time.perf_counter() - start < 1.0
    assert r.json() == {"detail": "The request took too long. Please try again.", "code": "timeout",
                        "request_id": r.headers["X-Request-ID"]}


def test_stream_over_the_cap_sends_timeout_event_and_saves_nothing(caplog):
    llm = FakeListChatModel(responses=["word " * 50], sleep=0.05)  # ~5 s of tokens
    client, _, _, memory = make(llm=llm, request_timeout_seconds=0.3)
    r = client.post("/api/chat/stream", json={"message": "What is acne?", "session_id": "s"})
    events = parse_sse(r.text)
    assert r.status_code == 200  # already streaming when the cap hit
    assert "token" in [e for e, _ in events]
    assert events[-1] == ("error", {"detail": "The request took too long. Please try again.",
                                    "code": "timeout", "request_id": r.headers["X-Request-ID"]})
    assert not asyncio.run(memory.exists("s"))
    assert events_named(caplog, "request_finished")[-1]["outcome"] == "timeout"


# --- Error mapping ------------------------------------------------------------------------


def test_bug_in_our_code_is_500_not_503():
    client, _, store, _ = make()

    def broken(text):
        raise RuntimeError("bug")

    store.embeddings.embed_query = broken
    r = client.post("/api/chat", json={"message": "What is acne?"})
    assert r.status_code == 500 and r.json()["code"] == "internal_error"


@pytest.mark.parametrize(
    "error",
    [
        httpx.ConnectError("refused"),
        httpx.ReadTimeout("slow"),
        openai.APITimeoutError(request=httpx.Request("POST", "https://api.openai.com")),
        ollama.ResponseError("overloaded", status_code=503),
    ],
)
def test_transient_llm_errors_are_503(error):
    llm = ScriptedLLM(responses=["x"], fail_calls=1, error=error)
    client, _, _, _ = make(llm=llm)
    r = client.post("/api/chat", json={"message": "What is acne?"})
    assert r.status_code == 503 and r.json()["code"] == "dependency_unavailable"


def test_non_transient_llm_error_is_500():
    llm = ScriptedLLM(responses=["x"], fail_calls=1, error=ollama.ResponseError("bad model", status_code=404))
    client, _, _, _ = make(llm=llm)
    assert client.post("/api/chat", json={"message": "What is acne?"}).status_code == 500


class StallingStreamLLM(FakeListChatModel):
    """Streams one token, then the provider times out (e.g. Ollama stops sending bytes)."""

    async def astream(self, input, config=None, **kwargs):
        async for chunk in super().astream(input, config, **kwargs):
            yield chunk
            raise httpx.ReadTimeout("no bytes for 30 s")


def test_stream_provider_timeout_mid_stream_is_dependency_event(caplog):
    client, _, _, memory = make(llm=StallingStreamLLM(responses=["Acne is a skin disease [1]."]))
    r = client.post("/api/chat/stream", json={"message": "What is acne?", "session_id": "s"})
    events = parse_sse(r.text)
    assert [e for e, _ in events].count("token") == 1
    assert events[-1][0] == "error" and events[-1][1]["code"] == "dependency_unavailable"
    assert events_named(caplog, "request_finished")[-1]["outcome"] == "dependency_unavailable"
    assert not asyncio.run(memory.exists("s"))  # partial answer not saved


def test_stream_bug_mid_stream_is_internal_error_event():
    llm = FakeListChatModel(responses=["Acne is a skin disease [1]."], error_on_chunk_number=3)
    client, _, _, _ = make(llm=llm)
    # FakeListChatModelError is not a provider error -> internal_error, not dependency_unavailable
    events = parse_sse(client.post("/api/chat/stream", json={"message": "What is acne?"}).text)
    assert events[-1][0] == "error" and events[-1][1]["code"] == "internal_error"


def test_404_and_422_bodies_have_code_and_request_id():
    client, _, _, _ = make()
    r = client.get("/api/sessions/missing/history")
    assert r.status_code == 404 and r.json()["code"] == "not_found" and r.json()["request_id"]
    r = client.post("/api/chat", json={"message": ""})
    assert r.status_code == 422 and r.json()["code"] == "validation_error" and r.json()["request_id"]
