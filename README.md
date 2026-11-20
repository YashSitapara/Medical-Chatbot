# Medical RAG Chatbot

A FastAPI service that answers medical questions from your PDFs (in `data/`). It gives:

- **numbered citations** with source file and page number
- **token streaming** over Server-Sent Events
- **per-session chat memory**, so follow-up questions work

Retrieval uses Pinecone with `all-MiniLM-L6-v2` embeddings. Generation uses a local Ollama model (default `gemma3:4b`) or OpenAI.

> For information only. This is not a substitute for professional medical advice.

## Project layout

```
app/
  main.py      FastAPI app, routes, SSE streaming
  config.py    settings from .env (pydantic-settings)
  schemas.py   request/response models
  rag.py       condense → retrieve → prompt → generate, citations
  memory.py    per-session chat history (in-process or Redis)
scripts/ingest.py   load PDFs → chunk → embed → upsert to Pinecone
frontend/index.html chat UI (served at /)
tests/              pytest suite (no network needed)
```

## Quick start (local)

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt
cp .env.example .env          # set PINECONE_API_KEY

# 1. Build the index (first time, or after changing PDFs/chunking)
python -m scripts.ingest --recreate

# 2. Start Ollama with the model (skip if LLM_PROVIDER=openai)
ollama pull gemma3:4b && ollama serve

# 3. Run the API + UI
uvicorn app.main:app --reload
```

Open http://localhost:8000 for the chat UI and http://localhost:8000/docs for the API docs.

`ingest` gives every chunk an id based on its file, page and position. Re-running it updates existing vectors and does not create duplicates. `--recreate` deletes the index and builds it again.

## Docker

```bash
cp .env.example .env
docker compose run --rm ingest --recreate   # one-off indexing job
docker compose up --build                   # nginx → API, on :8000
docker compose up --build --scale api=3     # 3 API replicas sharing history via Redis
```

Compose runs **Redis** (for chat history), the **API**, and **nginx** as the entry point on port 8000. nginx spreads requests across API replicas and passes streamed tokens through without buffering. Each response has an `X-Upstream` header showing which replica answered.

The API containers reach Ollama on your host at `host.docker.internal:11434`. The embedding model is downloaded when the image is built.

## API

| Method | Path | Description |
|---|---|---|
| `GET` | `/health` | Status, LLM provider/model, index name (503 if chat memory is down) |
| `GET` | `/ready` | Readiness: Redis ping + Pinecone index stats, each ≤ `READY_TIMEOUT_SECONDS` (5s). 200, or 503 naming the failed check |
| `POST` | `/api/chat` | `{message, session_id?}` → `{session_id, answer, citations[]}` |
| `POST` | `/api/chat/stream` | Same body, streamed back as SSE (see below) |
| `GET` | `/api/sessions/{id}/history` | Messages in the session (404 if unknown or expired) |
| `DELETE` | `/api/sessions/{id}` | Clear the session |

Leave out `session_id` to start a new conversation. The response includes the new id.

Each citation looks like this:

```json
{"id": 1, "source": "Medical_book.pdf", "page": 42, "snippet": "…", "score": 0.83, "cited": true}
```

`id` matches the `[n]` marker in the answer. `cited` is true when the answer actually used that source.

### Streaming events

```
event: session    data: {"session_id": "..."}
event: citations  data: [Citation, ...]              # sent before the first token
event: token      data: {"text": "..."}              # repeated
event: done       data: {"session_id", "answer", "citations"}
event: error      data: {"detail": "..."}            # instead of done, on failure
```

The turn is saved to memory only after `done`. If the client disconnects partway through, nothing is saved.

### Example

```bash
curl -s localhost:8000/api/chat -H 'content-type: application/json' \
  -d '{"message": "What is acne?"}'

curl -N localhost:8000/api/chat/stream -H 'content-type: application/json' \
  -d '{"message": "What causes it?", "session_id": "<id from above>"}'
```

## How a question is answered

1. **Condense.** If the session has history, the LLM rewrites the follow-up into a question that stands alone ("what causes it?" → "what causes acne?"). The search uses the rewritten question.
2. **Retrieve.** The top `RETRIEVER_K` chunks come from Pinecone, each with its similarity score.
   Chunks scoring below `MIN_RELEVANCE_SCORE` (default 0.40) are dropped. If none are left, the
   question is out of scope: the app answers "I couldn't find information about this…" with no
   citations and **does not call the LLM**. On this index, medical questions score about 0.48–0.76
   and unrelated ones about 0.18–0.32.
3. **Prompt.** The chunks are numbered `[1]…[k]` with file and page, then sent along with the chat history and the original question.
4. **Generate.** The LLM answers using only that context and cites passages as `[n]`. The answer is streamed back token by token.
5. **Remember.** The question/answer pair is stored. Each session keeps its last `MEMORY_MAX_TURNS` turns and is removed after `MEMORY_TTL_SECONDS` of inactivity.

Where history is stored depends on `MEMORY_BACKEND` (see [Scaling](#scaling-multiple-workers-or-containers)).

## Scaling: multiple workers or containers

With the default `MEMORY_BACKEND=memory`, history is a Python dict inside one process. Run several uvicorn workers (`--workers 4`) or several containers, and each has its **own** dict. A follow-up that lands on a different process loses its context, `/history` returns 404 at random, and "New chat" clears only one copy.

`MEMORY_BACKEND=redis` fixes this. Every process reads and writes the same Redis, so any worker can serve any request, and no sticky sessions are needed.

```bash
# Local, several workers (Redis runs in Docker, see "Local Redis (Docker)" below)
docker compose up -d redis
MEMORY_BACKEND=redis uvicorn app.main:app --workers 3

# Docker, several containers
docker compose up --build --scale api=3
```

### Local Redis (Docker)

You don't need to install Redis on your Mac. Any Docker runtime works (OrbStack, Docker Desktop or Colima), and `redis-cli` runs inside the container.

```bash
docker compose up -d redis                       # start; data is kept in the redis-data volume
docker compose exec redis redis-cli ping         # → PONG
docker compose exec redis redis-cli LRANGE medical-rag:chat:<id> 0 -1   # inspect a session
docker compose exec redis redis-cli TTL medical-rag:chat:<id>           # seconds until it expires
docker compose stop redis                        # stop (keeps data)
docker compose down -v                           # remove the containers AND the stored history
```

Redis is published on `127.0.0.1:6379` only, so the app on your Mac uses `REDIS_URL=redis://localhost:6379/0` (the default). Set `REDIS_PORT=6380` if 6379 is already taken, and change `REDIS_URL` to match.

For production, use a Redis with a password (`requirepass`, or a managed Redis) and TLS: `REDIS_URL=rediss://:password@host:6380/0`.

How it is stored:

- Each session is a Redis list at `medical-rag:chat:<session_id>`. Each item is one JSON message.
- **Saving a turn** is one transaction (`RPUSH` + `LTRIM` + `EXPIRE`). The user/assistant pair stays together even when two workers write at once, the list is trimmed to `MEMORY_MAX_TURNS`, and the expiry is reset to `MEMORY_TTL_SECONDS`.
- **Reading** history also resets the expiry, so active conversations never expire.
- To inspect a session: `docker compose exec redis redis-cli LRANGE medical-rag:chat:<id> 0 -1`.

If Redis is unreachable:

- the API **refuses to start**;
- while running, chat and history requests return **503** and `/health` returns 503, so load balancers and Docker take the instance out of rotation;
- it recovers by itself when Redis comes back.

### Health vs readiness

- `GET /health`: the process is up and chat memory answers.
- `GET /ready`: can this instance serve a chat right now? It runs a Redis `PING` and a Pinecone `describe_index_stats` call (no embedding, no query) **concurrently**, each limited to `READY_TIMEOUT_SECONDS` (default 5). It returns 200 when both pass, otherwise 503 with the failed check marked:

  ```json
  {"status": "not_ready", "checks": {
    "redis":    {"ok": false, "latency_ms": 3.1,   "error": "MemoryUnavailableError"},
    "pinecone": {"ok": true,  "latency_ms": 212.4, "error": null}}}
  ```

  `error` is only `timed out after Ns` or an exception class name; the full error is in the server log. With `MEMORY_BACKEND=memory` the first check is called `memory` and always passes.

The Docker `HEALTHCHECK` calls `/ready`. Docker then **marks** the container `unhealthy` (see `docker compose ps`), but plain Compose doesn't restart it, and nginx keeps sending it traffic. Use `/ready` as the readiness probe in Swarm or Kubernetes, which do act on it.

**Workers vs containers.** `WEB_CONCURRENCY` sets the uvicorn workers per container. Each worker loads its own copy of the embedding model, about 0.5 GB of RAM with torch, so add capacity with care.

## Configuration

See [.env.example](.env.example) for every setting. The main ones:

- `LLM_PROVIDER`: `ollama` or `openai`. `openai` needs `OPENAI_API_KEY`.
- `RETRIEVER_K`: how many chunks are retrieved per question.
- `MIN_RELEVANCE_SCORE`: the lowest cosine similarity a chunk needs to be used (0 disables the filter). This value depends on the embedding model.
- `EMBEDDING_MODEL`: used by both ingest and the API (one factory, `app/embeddings.py`). The vector dimension is measured from the model. If you change it, run `python -m scripts.ingest --recreate`; until then the API refuses to start, because the index dimension won't match.
- `MEMORY_MAX_TURNS`, `MEMORY_TTL_SECONDS`: how much history is kept and for how long.
- `MEMORY_BACKEND` (`memory` | `redis`), `REDIS_URL`, `REDIS_KEY_PREFIX`: where history is stored. Use `redis` for more than one worker or container.

## Tests

```bash
pytest -q
```

The tests use a fake LLM, a stub vector store and an in-process fake Redis (`fakeredis`). They need no Pinecone, Ollama, Redis or network access.

The memory tests run the same checks against every backend. To also run them against a real Redis server:

```bash
docker compose up -d redis
REDIS_URL=redis://localhost:6379/15 pytest -q -m redis
```

The tests use a unique key prefix and never touch existing keys.
