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
| `POST` | `/api/chat` | `{message, session_id?}` → `{session_id, answer, citations[]}`. Rate limited. |
| `POST` | `/api/chat/stream` | Same body, streamed back as SSE (see below). Rate limited. |
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
event: error      data: {"detail", "code", "request_id"}   # instead of done, on failure
```

The HTTP status is already 200 by the time a stream fails, so the `code` of the `error` event says what went wrong (same codes as below).

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
   If the rewrite fails, times out (`CONDENSE_TIMEOUT_SECONDS`) or looks like an answer, the request continues with a **dual search** instead: the plain follow-up *and* "previous question + follow-up" are searched in parallel and the chunks merged (each keeps its best score). A real follow-up ("who gets it?") wins through the combined query; a topic switch ("what causes migraines?" after diabetes) wins through the plain one. The logs record `condense_strategy`: `skipped`, `llm` or `fallback_dual`.
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

## Timeouts, retries and errors

A slow or failing dependency gives a fast, clear error instead of a stuck request.

| Step | Limit | Retries |
|---|---|---|
| Rewrite (condense) | `CONDENSE_TIMEOUT_SECONDS` = 10 | none: falls back to dual search, never fails the request |
| Pinecone query | `RETRIEVAL_TIMEOUT_SECONDS` = 5 per attempt (the HTTP call itself, not just the wait) | `RETRIEVAL_MAX_ATTEMPTS` = 3, exponential backoff with jitter, **only** for timeouts, connection errors and 5xx. A 4xx (bad key, missing index) means the request is wrong, so it is never retried. |
| LLM | `LLM_TIMEOUT_SECONDS` = 30 with no bytes received (long streamed answers are fine) | `LLM_MAX_RETRIES` = 2 (OpenAI), before the first token only |
| Whole request | `REQUEST_TIMEOUT_SECONDS` = 45 | — |

The steps' worst cases add up to more than 45 s; the overall cap is what bounds a request.

Every error response has the same body: `{"detail": "...", "code": "...", "request_id": "..."}`.

| Status | `code` | Meaning |
|---|---|---|
| 503 | `dependency_unavailable` | LLM, Pinecone or Redis is down, timed out or returned 5xx (after retries). Has `Retry-After: 5`. |
| 504 | `timeout` | The request hit `REQUEST_TIMEOUT_SECONDS`. Streams send an `error` event with this code instead. |
| 500 | `internal_error` | A bug in our code, or our configuration (e.g. the LLM or Pinecone rejects the API key). Not retried. |
| 429 | `rate_limited` | See below. |
| 404 / 422 | `not_found` / `validation_error` | Unknown session or route / invalid request body. |

A partial streamed answer is never saved to chat history.

## Rate limiting

`/api/chat` and `/api/chat/stream` are limited per minute (fixed window), so a single client can't run up the LLM bill. Other routes are not limited.

Each request counts against two counters in Redis:

- `ratelimit:session:{session_id}:{minute}`: `RATE_LIMIT_PER_MINUTE` (10), when the request has a session;
- `ratelimit:ip:{ip}:{minute}`: `RATE_LIMIT_PER_IP_PER_MINUTE` (30).

The per-IP counter exists because `session_id` comes from the client: a script could send a new one with every request. It is higher than the per-session limit so several people behind one NAT can still chat.

Each counter is `INCR` + `EXPIRE 60 NX` in one transaction. Over a limit: **429** with `Retry-After` (seconds left in the minute). Every chat response has `X-RateLimit-Limit` and `X-RateLimit-Remaining` for the counter closest to its limit.

- **Fixed window trade-off:** up to twice the limit can pass around a minute boundary (10 at 0:59, 10 at 1:00). Accepted for one `INCR` per counter and constant memory.
- **Fails open (deliberate):** if Redis is unreachable, requests are **allowed** and a `rate_limit_unavailable` warning is logged (at most every 30 s). The limiter protects cost; it is not a security boundary, and failing closed would turn a Redis blip into a full chat outage. The limiter uses its own Redis client with a 0.5 s timeout, so a slow Redis adds at most that.
- With `MEMORY_BACKEND=memory`, counters are kept in-process instead (per worker).
- `RATE_LIMIT_ENABLED=false` turns it off.

**Client IP behind a proxy.** The IP is `request.client.host`. uvicorn replaces it with the `X-Forwarded-For` address **only** when the direct peer is listed in `FORWARDED_ALLOW_IPS` (default `127.0.0.1`), so a client connecting directly can't fake its IP with that header. In Docker, nginx overwrites `X-Forwarded-For` with the real peer address, and the api service sets `FORWARDED_ALLOW_IPS="*"`, which is safe only because the api containers are reachable through nginx alone (`expose`, not `ports`). If you add a CDN or load balancer in front of nginx, configure nginx's `real_ip` module with that proxy's address ranges.

## Configuration

See [.env.example](.env.example) for every setting. The main ones:

- `LLM_PROVIDER`: `ollama` or `openai`. `openai` needs `OPENAI_API_KEY`.
- `RETRIEVER_K`: how many chunks are retrieved per question.
- `MIN_RELEVANCE_SCORE`: the lowest cosine similarity a chunk needs to be used (0 disables the filter). This value depends on the embedding model.
- `EMBEDDING_MODEL`: used by both ingest and the API (one factory, `app/embeddings.py`). The vector dimension is measured from the model. If you change it, run `python -m scripts.ingest --recreate`; until then the API refuses to start, because the index dimension won't match.
- `MEMORY_MAX_TURNS`, `MEMORY_TTL_SECONDS`: how much history is kept and for how long.
- `MEMORY_BACKEND` (`memory` | `redis`), `REDIS_URL`, `REDIS_KEY_PREFIX`: where history is stored. Use `redis` for more than one worker or container.
- `LOG_LEVEL`, `LOG_FORMAT` (`json` | `console`), `LOG_CONTENT`: see [Logging](#logging).
- Timeouts and retries: see [Timeouts, retries and errors](#timeouts-retries-and-errors). Rate limits: see [Rate limiting](#rate-limiting).

## Logging

Logs are JSON, one object per line on stdout (`LOG_FORMAT=console` gives readable output in a terminal). Every line has `timestamp`, `level`, `logger`, `pid`, `event`, and, during a request, `request_id` and `session_id`.

**Request IDs.** Each request gets an ID. If the client sends a header `X-Request-ID` made of letters, digits and `. _ : -` (at most 128 characters), that ID is reused; anything else is replaced, so a client can't forge log lines. The ID is returned in the `X-Request-ID` response header, including on errors, and in SSE `error` events; the frontend shows its first 8 characters as `ref:`. nginx creates or forwards the same ID, so its access log matches the app's lines.

**A chat request logs these events:**

| event | fields |
|---|---|
| `request_started` | `route`, `method`, `stream`, `new_session`, `question_chars` |
| `query_condensed` | `condense_strategy` (`skipped` / `llm` / `fallback_dual`), `changed`, `reason` (fallback only), `history_messages`, `latency_ms` |
| `retrieval_done` | `query_kind` (`standalone` / `plain` / `combined`), `chunks_retrieved`, `chunks_passed` (at or above the threshold), `min_score`, `top_score`, `scores`, `latency_ms` |
| `llm_done` | `provider`, `model`, `input_tokens`, `output_tokens`, `latency_ms`, `ttft_ms` (streaming). Absent when the question is out of scope. |
| `request_finished` | `route` (template, e.g. `/api/sessions/{session_id}/history`), `method`, `status_code`, `latency_ms` (total, including the whole stream), `outcome` |

Every request logs `request_finished`. `outcome` is one of `ok`, `no_context`, `client_error` (4xx), `rate_limited`, `dependency_unavailable`, `timeout`, `error`, `client_disconnected` or `exception`. It matters for streaming: a stream that fails part-way has already sent status 200. Successful `/health` and `/ready` calls are logged at DEBUG, so Docker's probes don't flood the logs. A `rewrite_rejected` warning (with a `reason`) shows when the follow-up rewrite fell back, `retrieval_retry` shows each Pinecone retry, and `rate_limited` / `dependency_unavailable` / `request_timeout` explain 429, 503 and 504 responses.

**Following one request:**

```bash
uvicorn app.main:app 2>&1 | tee app.log
jq -c 'select(.request_id == "9f2c...")' app.log
jq -c 'select(.event == "llm_done") | {latency_ms, input_tokens, output_tokens}' app.log
```

**Privacy.** Logs carry metadata only: lengths, scores, counts, latencies.

- User questions, rewritten questions and answers are **never** logged. As a safety net, each request's question and rewrite are replaced with `[question]` wherever else they show up, such as in an exception message.
- `LOG_CONTENT=true` adds `question` (to `request_started`) and `condensed_question` (to `query_condensed`). It is for **local development only**, and startup logs a warning when it's on. Answers are still not logged.
- API keys are always redacted, in our logs and in third-party ones (uvicorn, redis, langchain, ...), including tracebacks. That covers the configured Pinecone/OpenAI keys, `pcsk_…`/`sk-…` keys, `Bearer` tokens, passwords in URLs, and fields named like `api_key`, `password`, `secret` or `token`.

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
