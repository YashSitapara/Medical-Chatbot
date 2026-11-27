# Medical RAG Chatbot

A FastAPI service that answers medical questions from your PDFs (in `data/`). It gives:

- **numbered citations** with source file and page number
- **token streaming** over Server-Sent Events
- **per-session chat memory**, so follow-up questions work

Retrieval uses Pinecone with `all-MiniLM-L6-v2` embeddings. Generation uses a local Ollama model (default `gemma3:4b`) or OpenAI.

> For information only. This is not a substitute for professional medical advice.

Release history: see [CHANGELOG.md](CHANGELOG.md) (current version: `app/__init__.py`, also shown at `/docs`).

## Project layout

```
app/
  main.py      FastAPI app, routes, SSE streaming
  config.py    settings from .env (pydantic-settings)
  schemas.py   request/response models
  rag.py       condense → retrieve → prompt → generate, citations
  memory.py    per-session chat history (in-process or Redis)
  tracing.py   Langfuse traces + feedback scores, PII masking
scripts/ingest.py   load PDFs → chunk → embed → upsert to Pinecone
scripts/langfuse_retention.py   delete Langfuse traces older than 30 days (run by hand)
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

**Ollama in Docker.** The API containers reach Ollama on your machine at `http://host.docker.internal:11434`. They deliberately ignore `OLLAMA_BASE_URL` from `.env`, which is `http://localhost:11434` for running uvicorn on the host: inside a container, `localhost` is the container itself, so every answer would fail with "The assistant is temporarily unavailable" (`ConnectError` in the logs). To use a different Ollama, such as a GPU server or the optional `ollama` service, set `DOCKER_OLLAMA_BASE_URL`:

```bash
DOCKER_OLLAMA_BASE_URL=http://my-gpu-box:11434 docker compose up -d   # or put it in .env
docker compose exec api printenv OLLAMA_BASE_URL                      # what the containers actually use
```

`/ready` checks Redis and Pinecone, not the LLM, so a wrong Ollama URL still shows the containers as healthy.

The API containers reach Ollama on your host at `host.docker.internal:11434`. The embedding model is downloaded when the image is built.

## API

| Method | Path | Description |
|---|---|---|
| `GET` | `/health` | Status, LLM provider/model, index name (503 if chat memory is down) |
| `GET` | `/ready` | Readiness: Redis ping + Pinecone index stats, each ≤ `READY_TIMEOUT_SECONDS` (5s). 200, or 503 naming the failed check |
| `POST` | `/api/chat` | `{message, session_id?}` → `{session_id, answer, citations[], trace_id}`. Rate limited. |
| `POST` | `/api/chat/stream` | Same body, streamed back as SSE (see below). Rate limited. |
| `POST` | `/api/feedback` | `{trace_id, rating: 1 \| -1, comment?}` → 202. 👍/👎 on an answer, stored on its Langfuse trace. 503 `tracing_disabled` when tracing is off. Rate limited (own counter). |
| `GET` | `/api/sessions/{id}/history` | Messages in the session (404 if unknown or expired, 422 if `id` isn't a UUID4) |
| `DELETE` | `/api/sessions/{id}` | Clear the session (422 if `id` isn't a UUID4) |

Leave out `session_id` to start a new conversation. The response includes the new id, a random UUID4 issued by the server. Send that id back for follow-ups. Any other value (`"test"`, `"user1"`, an upper-case or non-v4 UUID) is rejected with **422** `validation_error`, because session ids act as bearer tokens (see [Privacy](#privacy-session-ids)).

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
event: done       data: {"session_id", "answer", "citations", "trace_id"}
event: error      data: {"detail", "code", "request_id", "trace_id"}   # instead of done, on failure
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
| 503 | `tracing_disabled` / `feedback_unavailable` | `POST /api/feedback` only: tracing is off, or the score could not be queued. |
| 404 / 422 | `not_found` / `validation_error` | Unknown session or route / invalid request body or a session id that isn't a server-issued UUID4. |

A partial streamed answer is never saved to chat history.

## Rate limiting

`/api/chat` and `/api/chat/stream` are limited per minute (fixed window), so a single client can't run up the LLM bill. Other routes are not limited.

Each request counts against two counters in Redis:

- `ratelimit:session:{session_id}:{minute}`: `RATE_LIMIT_PER_MINUTE` (10), when the request has a session;
- `ratelimit:ip:{ip}:{minute}`: `RATE_LIMIT_PER_IP_PER_MINUTE` (30).

The per-IP counter exists because `session_id` comes from the client: it must be UUID-shaped, but a script could still send a new random one with every request. It is higher than the per-session limit so several people behind one NAT can still chat.

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
- `TRACING_ENABLED`, `LANGFUSE_*`: see [Tracing & feedback](#tracing--feedback-langfuse).
- `CORS_ORIGINS`: browser origins allowed to call the API cross-origin. **Default: none.** The bundled UI is served from the same origin and needs nothing. Add origins explicitly, e.g. `CORS_ORIGINS=["http://localhost:3000"]`. Avoid `["*"]`: session ids are bearer tokens.

## Tracing & feedback (Langfuse)

Logs deliberately contain no text, so they can't show *what the model was given*. A Langfuse trace can: for any answer, open one trace and see the rewritten question, the chunks retrieved (with scores and pages), the exact prompt, and the output with token counts. Users rate answers with 👍/👎, and the rating is attached to that trace.

### Setup (Langfuse Cloud, free Hobby tier)

1. Sign up at [cloud.langfuse.com](https://cloud.langfuse.com) (EU region; recommended, since traces contain health questions) and create a project.
2. In **Project Settings → API Keys**, create a key pair and add it to `.env`:
   ```bash
   TRACING_ENABLED=true
   LANGFUSE_PUBLIC_KEY=pk-lf-...
   LANGFUSE_SECRET_KEY=sk-lf-...
   # LANGFUSE_BASE_URL=https://us.cloud.langfuse.com   # only for the US region
   ```
3. Restart the app. The startup log shows `tracing_ready` with `enabled: true`.

With `TRACING_ENABLED=true` and a key missing, the app refuses to start. With tracing off (the default), responses carry `trace_id: null` and the 👍/👎 buttons are hidden.

### What a trace contains

One trace per chat request, named `chat` or `chat-stream`, grouped by session (**Sessions** tab):

| Observation | Input | Output / details |
|---|---|---|
| `chat` (root) | the question | answer and the cited passage numbers; `outcome`, `condense_strategy`; level ERROR/WARNING on failures (exception class name only) |
| `condense` | the chat transcript actually sent, and the question | rewritten question, `strategy` (`skipped` / `llm` / `fallback_dual`), rejection `reason`, `raw_rewrite`; model and tokens. WARNING when the rewrite was rejected. |
| `retrieve` | the search query (or both fallback queries) | every chunk returned: `id`, `score`, `page`, `source`, `passed` (above the threshold, i.e. used); `attempts` (Pinecone retries) |
| `retrieve.plain` / `retrieve.combined` | one fallback query each | that search's chunks (condense fallback only) |
| `generate` | the full prompt as messages (system prompt with numbered context, history, question) | the model's raw answer, model, temperature, input/output tokens, time to first token (streaming). WARNING `cancelled` if the stream was cut off. |

**Trace fields:** session ID, `user_id` (empty until the app has users), tags (LLM provider), version (= prompt version), and metadata: `request_id`, `prompt_version`, `prompt_sha`, `llm_provider`, `llm_model`, `embedding_model`, `index`, `k`, `min_score`. `request_id` matches the logs, and every log line of a traced request also carries `trace_id`, so you can go from either one to the other.

Bump `PROMPT_VERSION` in `app/rag.py` when you change a prompt. `prompt_sha` changes automatically, so a forgotten bump still shows.

**Debugging a bad answer:** open the trace (filter by `request_id` in metadata, or by session), then check:
- `condense`: was the question rewritten correctly?
- `retrieve`: were the right chunks found, above the threshold?
- `generate`: what exactly did the model receive?

Tracing never slows down or fails a request. Spans are sent in the background, and a Langfuse outage only produces warnings in the logs.

### Feedback

Each answer with a `trace_id` gets 👍 / 👎 buttons, and after a click an optional comment box (up to 500 characters). The frontend calls `POST /api/feedback`, which stores a `user_feedback` score (`1` or `-1`, with the comment) on the trace. You can find it in the **Scores** tab or filter traces by it.

- **Changing your mind:** rating the same answer again replaces its score. The score ID is derived from the trace ID, so there's never a duplicate.
- **Restored history:** `trace_id` is saved with each answer, so answers in restored history can still be rated.
- **Who can rate a trace:** there is no login yet, so possessing the `trace_id` is the permission. It's 128 random bits and only ever sent to the client that asked the question. Once there are users, ownership can be checked against `user_id`.
- **Rate limit:** feedback has its own per-IP counter (`FEEDBACK_RATE_LIMIT_PER_IP_PER_MINUTE`, default 30), so rating answers never uses up the chat allowance.

### Privacy: session IDs

**Session IDs are unguessable and act as bearer tokens.** The server issues a random UUID4 for each new conversation, and the API rejects any other value with 422, in the chat body and in `/api/sessions/{id}/…`. A client therefore can't pick a guessable id like `user1` that someone else could read. Anyone who holds a session id can read and delete that conversation, so treat it like a password: the UI keeps it in the browser's `localStorage`, and it shouldn't be pasted into shared places. `trace_id` works the same way for `POST /api/feedback`.

### Privacy: what is sent to Langfuse

Traces contain **text**: questions, chat history, the prompt and the answer. Before anything leaves the app, `mask_pii` in `app/tracing.py` replaces:

| Pattern | Becomes |
|---|---|
| email addresses | `[EMAIL]` |
| Indian mobile numbers (`+91 98765 43210`, `09876543210`, `9876543210`) and international numbers starting with `+` | `[PHONE]` |
| Aadhaar numbers (12 digits, optionally `1234 5678 9012`) | `[AADHAAR]` |
| PAN (`ABCDE1234F`, any case) | `[PAN]` |

It applies to every observation's input, output and metadata, and to feedback comments. Numbers such as doses, pages, years and scores are left alone. If the masking code itself ever fails, the SDK sends a placeholder instead of the value, so a bug can't leak unmasked text.

**Masking is pattern-based and does not catch everything.** Names, addresses and free-text health details are still sent. Langfuse is therefore a processor of users' health questions. Use the EU region, keep retention short (below), and leave `TRACING_ENABLED=false` wherever that isn't acceptable. Tracing is independent of `LOG_CONTENT`: the logs still never contain text.

### Retention policy

**Traces and their feedback scores are kept for 30 days, then deleted.**

Langfuse's built-in **Data Retention** setting is only on paid plans (Pro/Enterprise) and self-hosted Enterprise. On the free Hobby plan, data older than 30 days is only *hidden* in the UI, not deleted. Until the deletion is scheduled, run the retention script by hand, at least weekly:

```bash
# 1. Dry run: counts the traces older than LANGFUSE_RETENTION_DAYS (30), deletes nothing
python -m scripts.langfuse_retention

# 2. Delete them (permanent)
python -m scripts.langfuse_retention --yes

# Other period, e.g. 14 days
python -m scripts.langfuse_retention --days 14 --yes

# In Docker
docker compose run --rm --entrypoint python api -m scripts.langfuse_retention --yes
```

The script finds traces whose root observation started before the cutoff and deletes them through the Langfuse API in batches of 100. It prints counts only, never content. Langfuse deletes the data in the background, so the traces can take a few minutes to disappear from the UI.

On a paid plan, set **Project Settings → Data Retention** to 30 days instead, and stop running the script.

## Logging

Logs are JSON, one object per line on stdout (`LOG_FORMAT=console` gives readable output in a terminal). Every line has `timestamp`, `level`, `logger`, `pid`, `event`, and, during a request, `request_id`, `session_id` and (when tracing is on) `trace_id`.

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
- API keys are always redacted, in our logs and in third-party ones (uvicorn, redis, langchain, ...), including tracebacks. That covers the configured Pinecone/OpenAI/Langfuse keys, `pcsk_…`/`sk-…`/`pk-lf-…` keys, `Bearer` tokens, passwords in URLs, and fields named like `api_key`, `password`, `secret` or `token`.

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

To check each feature by hand on the real stack (several workers/containers, request IDs, rate limits, Langfuse tracing and feedback), follow [NOTES.md](NOTES.md).
