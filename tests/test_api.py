import json

from tests.conftest import run


def parse_sse(text: str) -> list[tuple[str, object]]:
    events = []
    for block in text.strip().split("\n\n"):
        lines = dict(line.split(": ", 1) for line in block.splitlines())
        events.append((lines["event"], json.loads(lines["data"])))
    return events


def test_health(make_client):
    client, _, _ = make_client([])
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {
        "status": "ok", "llm_provider": "ollama", "llm_model": "gemma3:4b",
        "index": "medical-chatbot", "memory_backend": "memory",
    }


def test_index_serves_frontend(make_client):
    client, _, _ = make_client([])
    r = client.get("/")
    assert r.status_code == 200 and "Medical Chatbot" in r.text


def test_chat_returns_answer_citations_and_session(make_client):
    client, _, memory = make_client(["Acne is a skin disease [1]."])
    r = client.post("/api/chat", json={"message": "What is acne?"})
    assert r.status_code == 200
    body = r.json()
    assert body["answer"] == "Acne is a skin disease [1]."
    assert body["session_id"]
    assert body["citations"][0] == {
        "id": 1, "source": "Medical_book.pdf", "page": 12,
        "snippet": "Acne is a common skin disease.", "score": 0.91, "cited": True,
    }
    assert len(run(memory.get(body["session_id"]))) == 2


def test_followup_uses_history(make_client):
    client, store, _ = make_client(["A skin disease [1].", "What causes acne?", "Clogged pores [2]."])
    sid = client.post("/api/chat", json={"message": "What is acne?"}).json()["session_id"]
    r = client.post("/api/chat", json={"message": "What causes it?", "session_id": sid})
    assert r.json()["answer"] == "Clogged pores [2]."
    assert store.queries == ["What is acne?", "What causes acne?"]

    hist = client.get(f"/api/sessions/{sid}/history").json()
    assert [m["content"] for m in hist["messages"]] == [
        "What is acne?", "A skin disease [1].", "What causes it?", "Clogged pores [2].",
    ]


def test_validation_errors(make_client):
    client, _, _ = make_client([])
    assert client.post("/api/chat", json={"message": "   "}).status_code == 422
    assert client.post("/api/chat", json={}).status_code == 422
    assert client.post("/api/chat", json={"message": "x" * 2001}).status_code == 422


def test_stream_emits_events_in_order_and_saves_history(make_client):
    client, _, memory = make_client(["Acne [1]."])
    with client.stream("POST", "/api/chat/stream", json={"message": "What is acne?"}) as r:
        assert r.status_code == 200
        assert r.headers["content-type"].startswith("text/event-stream")
        events = parse_sse(r.read().decode())

    names = [e for e, _ in events]
    assert names[0] == "session" and names[1] == "citations" and names[-1] == "done"
    assert set(names[2:-1]) == {"token"}
    assert "".join(d["text"] for e, d in events if e == "token") == "Acne [1]."
    sid = events[0][1]["session_id"]
    done = events[-1][1]
    assert done["answer"] == "Acne [1]." and done["session_id"] == sid
    assert done["citations"][0]["cited"] is True
    assert len(run(memory.get(sid))) == 2


def _pinecone_down(store, times=3):
    from pinecone.exceptions import ServiceException

    store.index.query_errors = [ServiceException(status=503, reason="unavailable") for _ in range(times)]


def test_stream_error_event(make_client):
    client, store, memory = make_client(["unused"])
    _pinecone_down(store)
    with client.stream("POST", "/api/chat/stream", json={"message": "q", "session_id": "s1"}) as r:
        events = parse_sse(r.read().decode())
    assert [e for e, _ in events] == ["session", "error"]
    assert events[-1][1]["code"] == "dependency_unavailable" and events[-1][1]["request_id"]
    assert not run(memory.exists("s1"))


def test_chat_returns_503_on_backend_failure(make_client):
    client, store, _ = make_client(["unused"])
    _pinecone_down(store)
    r = client.post("/api/chat", json={"message": "q"})
    assert r.status_code == 503
    assert r.json()["code"] == "dependency_unavailable" and r.headers["Retry-After"] == "5"


def test_history_404_and_delete(make_client):
    client, _, _ = make_client(["A [1]."])
    assert client.get("/api/sessions/missing/history").status_code == 404
    sid = client.post("/api/chat", json={"message": "q"}).json()["session_id"]
    assert client.delete(f"/api/sessions/{sid}").status_code == 204
    assert client.get(f"/api/sessions/{sid}/history").status_code == 404


def test_chat_out_of_scope(make_client):
    from app.rag import NO_CONTEXT_ANSWER
    from tests.conftest import OFF_TOPIC

    client, _, _ = make_client([], results=OFF_TOPIC)
    body = client.post("/api/chat", json={"message": "best gaming laptop?"}).json()
    assert body["answer"] == NO_CONTEXT_ANSWER and body["citations"] == []

    with client.stream("POST", "/api/chat/stream", json={"message": "best gaming laptop?"}) as r:
        events = parse_sse(r.read().decode())
    assert [e for e, _ in events] == ["session", "citations", "token", "done"]
    assert events[1][1] == [] and events[-1][1]["answer"] == NO_CONTEXT_ANSWER


def test_stream_done_and_memory_use_normalized_citations(make_client):
    client, _, memory = make_client(["Acne [1, 2]."])
    with client.stream("POST", "/api/chat/stream", json={"message": "What is acne?"}) as r:
        events = parse_sse(r.read().decode())
    done = events[-1][1]
    assert done["answer"] == "Acne [1][2]."
    assert [c["cited"] for c in done["citations"]][:2] == [True, True]
    stored = run(memory.get(events[0][1]["session_id"]))
    assert stored[-1].content == "Acne [1][2]."
