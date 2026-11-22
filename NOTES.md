# Manual verification guide

Copy-paste commands to check four features on the **real** stack (Ollama, Pinecone, Redis):

1. [Redis with multiple workers / containers](#1-redis-with-multiple-workers-and-containers)
2. [Structured logging with request IDs](#2-structured-logging-with-request-ids)
3. [Rate limiting with Redis](#3-rate-limiting-with-redis)
4. [Langfuse tracing and feedback](#4-langfuse-tracing-and-feedback)

The automated tests cover the same behaviour with fakes, so run them first:

```bash
pytest -q                                          # no network needed
REDIS_URL=redis://localhost:6379/15 pytest -q      # + the tests that need a real Redis (DB 15)
```

Each section ends with an **Expected** checklist. Every session ID used here starts with `verify-`, so [cleanup](#5-cleanup) can't touch real conversations.

---

## 0. Setup (once per terminal)

```bash
cd ~/Projects/Medical-Chatbot
source .venv/bin/activate
export PATH="$HOME/.orbstack/bin:$PATH"            # docker (OrbStack)

docker compose up -d redis
docker compose exec redis redis-cli ping           # → PONG
curl -s localhost:11434/api/tags | jq -r '.models[].name'   # Ollama is up and lists gemma3:4b
```

There is no `redis-cli` on the Mac. It runs inside the container, so this shortcut helps:

```bash
rcli() { docker compose exec -T redis redis-cli "$@"; }
```

Choose where requests go. The Docker stack (nginx) listens on **8000**; local uvicorn runs below use **8001**, so both can run at once.

```bash
export BASE=http://localhost:8000     # Docker (nginx -> api replicas)
# export BASE=http://localhost:8001   # local uvicorn
```

Helpers (zsh/bash). Each prints the status line, the interesting headers, then the JSON body:

```bash
# chat "<question>" [session_id] [extra curl args...]
chat() {
  local q="$1" sid="$2"; if [ $# -ge 2 ]; then shift 2; else shift $#; fi
  local body; body=$(jq -nc --arg m "$q" --arg s "$sid" '{message:$m} + (if $s == "" then {} else {session_id:$s} end)')
  curl -s -D /tmp/verify-headers -o /tmp/verify-body -H 'content-type: application/json' "$@" "$BASE/api/chat" -d "$body"
  head -1 /tmp/verify-headers | tr -d '\r'
  grep -iE '^(x-request-id|x-upstream|x-ratelimit-[a-z]+|retry-after):' /tmp/verify-headers | tr -d '\r'
  jq -c '{answer: (.answer // "" | .[0:100]), trace_id, code, detail, request_id}' /tmp/verify-body
}

# stream "<question>" [session_id]: prints each SSE event name + a short payload
stream() {
  local body; body=$(jq -nc --arg m "$1" --arg s "$2" '{message:$m} + (if $s == "" then {} else {session_id:$s} end)')
  curl -sN "$BASE/api/chat/stream" -H 'content-type: application/json' -d "$body" \
    | awk '/^event:/{e=$2} /^data:/{sub(/^data: /,""); print e": "substr($0,1,140)}'
}
```

Where the logs are:

| Running as | Logs |
|---|---|
| Docker | `docker compose logs api --no-log-prefix` (JSON lines; add `-f` to follow) |
| Local uvicorn | the file you `tee` into, e.g. `/tmp/app.log` |

> **Docker and Ollama.** The containers reach Ollama at `http://host.docker.internal:11434`, never at `.env`'s `OLLAMA_BASE_URL`, which is `localhost` for the host. Check it with `docker compose exec api printenv OLLAMA_BASE_URL`. If every answer says *"The assistant is temporarily unavailable"* and the logs show `ConnectError`, this URL is wrong. `/ready` doesn't check the LLM, so the containers still look healthy.

---

## 1. Redis with multiple workers and containers

**What should happen:** with `MEMORY_BACKEND=redis`, every worker and container reads and writes the same history in Redis. A follow-up question works no matter which process answers it.

### 1a. Docker: 3 replicas behind nginx

```bash
docker compose up --build -d --scale api=3        # (no --build needed if the image is current)
docker compose ps                                  # api-1..3 "(healthy)", nginx, redis
docker compose exec api printenv MEMORY_BACKEND OLLAMA_BASE_URL   # redis, http://host.docker.internal:11434
export BASE=http://localhost:8000
```

**Requests are spread over the replicas.** Each response's `X-Upstream` names the replica that answered:

```bash
for i in 1 2 3 4 5 6; do curl -s -o /dev/null -D - $BASE/health | grep -i x-upstream; done
# → 2-3 different addresses, e.g. 192.168.97.3:8000, 192.168.97.4:8000, 192.168.97.5:8000
```

**One conversation, answered by different replicas:**

```bash
SID=verify-docker-1
chat "What is acne?" $SID
chat "Who usually gets it?" $SID          # a follow-up: needs the first turn from Redis
chat "How is it treated?" $SID
```

**Each of those requests went to a different replica, yet the follow-ups knew what "it" meant:**

```bash
docker compose logs api --no-log-prefix | jq -c 'select(.session_id=="'$SID'" and .event=="query_condensed")
  | {request_id: .request_id[0:8], condense_strategy, changed}'
# → turn 1: "skipped"; turns 2-3: "llm", changed: true  (the rewrite saw the history)

docker compose logs api | grep '"session_id": "'$SID'"' | grep request_finished | cut -d'|' -f1 | sort | uniq -c
# → the lines come from more than one container (api-1, api-2, api-3)
```

**Every replica returns the same history:**

```bash
for i in 1 2 3 4 5 6; do
  curl -s -D /tmp/h $BASE/api/sessions/$SID/history | jq -c '[.messages | length]' | tr -d '\n'
  echo "  via $(grep -i x-upstream /tmp/h | tr -d '\r' | cut -d' ' -f2)"
done
# → [6] from every replica
```

**The history is stored in Redis:**

```bash
rcli LLEN medical-rag:chat:$SID           # → 6  (3 turns = 6 messages)
rcli TTL  medical-rag:chat:$SID           # → ≤ 3600 (MEMORY_TTL_SECONDS); reset on every read/write
rcli LRANGE medical-rag:chat:$SID 0 -1 | head -2
```

**"New chat" clears it everywhere:**

```bash
curl -s -o /dev/null -w '%{http_code}\n' -X DELETE $BASE/api/sessions/$SID      # → 204
for i in 1 2 3; do curl -s -o /dev/null -w '%{http_code} ' $BASE/api/sessions/$SID/history; done; echo
# → 404 404 404 (no replica still has a copy)
```

### 1b. Local: several uvicorn workers in one process group

```bash
export BASE=http://localhost:8001
MEMORY_BACKEND=redis uvicorn app.main:app --workers 3 --port 8001 2>&1 | tee /tmp/app.log
```

In a second terminal (run the [setup helpers](#0-setup-once-per-terminal) again there, with `BASE=http://localhost:8001`):

```bash
SID=verify-workers-1
chat "What is acne?" $SID
for q in "Who usually gets it?" "How is it treated?" "Can it leave scars?" "Is it contagious?"; do chat "$q" $SID; done

# Which worker (pid) served each request:
jq -c 'select(.session_id=="'$SID'" and .event=="request_finished") | {pid, status_code}' /tmp/app.log
# → several different pids, all 200

# Every follow-up still used the history:
jq -c 'select(.session_id=="'$SID'" and .event=="query_condensed") | {pid, condense_strategy, changed}' /tmp/app.log

# History is the same whichever worker answers:
for i in $(seq 6); do curl -s $BASE/api/sessions/$SID/history | jq '.messages | length'; done | sort | uniq -c
# → "6 10" (6 calls, all 10 messages)
```

**Contrast: what goes wrong without Redis.** Stop the server (Ctrl-C) and start it with in-process memory:

```bash
MEMORY_BACKEND=memory uvicorn app.main:app --workers 3 --port 8001 2>&1 | tee /tmp/app-memory.log
grep inmemory_with_multiple_workers /tmp/app-memory.log     # startup warning, one per worker

SID=verify-workers-2
chat "What is acne?" $SID
for i in $(seq 9); do curl -s -o /dev/null -w '%{http_code} ' $BASE/api/sessions/$SID/history; done; echo
# → a mix of 200 and 404: only the worker that answered has the history
```

Stop it and go back to `MEMORY_BACKEND=redis` for the next sections.

### 1c. Redis goes away

```bash
export BASE=http://localhost:8000
docker compose pause redis
chat "What is acne?" verify-down-1               # → 503, code: dependency_unavailable (after ~3-6 s)
curl -s $BASE/ready | jq -c '.status, .checks.redis'   # → "not_ready", {"ok": false, "error": "..."}
docker compose unpause redis
chat "What is acne?" verify-down-1               # → 200 again, nothing restarted
```

The Docker healthcheck marks replicas `unhealthy` only after 3 failed `/ready` probes, 30 s apart. A short pause won't show up in `docker compose ps`.

### Expected

- [ ] `X-Upstream` shows several replicas.
- [ ] The follow-ups in one session were served by different replicas or pids and still have `condense_strategy: "llm"`.
- [ ] `/history` returns the same message count from every replica or worker; `LLEN` matches.
- [ ] `DELETE` gives 404 everywhere afterwards.
- [ ] With `MEMORY_BACKEND=memory` and 3 workers, `/history` flips between 200 and 404, and the startup warning appears.
- [ ] With Redis paused: 503 `dependency_unavailable` and `/ready` 503 naming `redis`. After unpausing, it recovers without a restart.

---

## 2. Structured logging with request IDs

**What should happen:**
- One JSON object per log line.
- Each request has an ID: reused from the client's `X-Request-ID` if it's valid, otherwise generated. It's returned in the response and printed on every log line of that request.
- No question text and no keys appear in the logs.

Use the local server for this, so the log is a single file:

```bash
export BASE=http://localhost:8001
MEMORY_BACKEND=redis LOG_FORMAT=json uvicorn app.main:app --port 8001 2>&1 | tee /tmp/app.log
```

### 2a. Request IDs

```bash
# Generated when you don't send one (32 hex characters):
curl -s -o /dev/null -D - $BASE/health | grep -i x-request-id

# Reused when you send a valid one:
curl -s -o /dev/null -D - -H 'X-Request-ID: verify-req-123' $BASE/health | grep -i x-request-id
# → x-request-id: verify-req-123

# Replaced when it's unsafe (spaces, >128 chars, ...): a client can't forge log lines
curl -s -o /dev/null -D - -H 'X-Request-ID: bad id with spaces' $BASE/health | grep -i x-request-id
# → a fresh 32-hex id

# Error bodies carry the same ID as the header:
curl -s -D - -H 'X-Request-ID: verify-404' $BASE/api/sessions/verify-nope/history | grep -iE 'x-request-id|request_id'
# → header verify-404, body {"detail":"Session not found","code":"not_found","request_id":"verify-404"}
curl -s -H 'X-Request-ID: verify-422' -H 'content-type: application/json' $BASE/api/chat -d '{}' | jq -c '{code, request_id}'
# → {"code":"validation_error","request_id":"verify-422"}
```

### 2b. Following one request through the logs

```bash
chat "What is acne?" verify-log-1 -H 'X-Request-ID: verify-chat-1'
chat "Who usually gets it?" verify-log-1 -H 'X-Request-ID: verify-chat-2'

jq -c 'select(.request_id=="verify-chat-2") | {event, latency_ms, condense_strategy, changed,
       chunks_retrieved, chunks_passed, top_score, input_tokens, output_tokens, status_code, outcome}
       | with_entries(select(.value != null))' /tmp/app.log
```

**Expected sequence, all sharing `request_id` and `session_id`:**

| event | look for |
|---|---|
| `request_started` | `route`, `stream`, `question_chars` (the length, never the text) |
| `query_condensed` | `condense_strategy` (`skipped` on turn 1, `llm` on follow-ups), `changed`, `latency_ms` |
| `retrieval_done` | `chunks_retrieved`, `chunks_passed`, `top_score`, `scores`, `latency_ms` |
| `llm_done` | `model`, `input_tokens`, `output_tokens`, `latency_ms` |
| `request_finished` | `status_code`, total `latency_ms`, `outcome: "ok"` |

**Streaming errors carry the ID too.** Restart with a tiny overall cap:

```bash
MEMORY_BACKEND=redis REQUEST_TIMEOUT_SECONDS=1 uvicorn app.main:app --port 8001 2>&1 | tee /tmp/app.log
stream "Explain acne in detail" verify-log-2
# → ... error: {"detail": "The request took too long...", "code": "timeout", "request_id": "...", ...}
jq -c 'select(.event=="request_finished") | {status_code, outcome}' /tmp/app.log | tail -1
# → {"status_code":200,"outcome":"timeout"}   (200 was already sent; outcome tells the truth)
```

Restart without `REQUEST_TIMEOUT_SECONDS` afterwards.

### 2c. Privacy

```bash
grep -c "What is acne" /tmp/app.log            # → 0  (question text is never logged)
grep -c "Who usually gets it" /tmp/app.log     # → 0
grep -cE "pcsk_[A-Za-z0-9]{6}" /tmp/app.log    # → 0  (Pinecone key never appears)
jq -c 'select(.event=="request_started") | {question_chars, question}' /tmp/app.log | tail -1
# → {"question_chars":13,"question":null}
```

**`LOG_CONTENT=true` (local debugging only) adds the text:**

```bash
MEMORY_BACKEND=redis LOG_CONTENT=true uvicorn app.main:app --port 8001 2>&1 | tee /tmp/app-content.log
grep log_content_enabled /tmp/app-content.log                  # startup warning
chat "What is acne?" verify-log-3
jq -c 'select(.event=="request_started" or .event=="query_condensed") | {event, question, condensed_question}' /tmp/app-content.log
# → question / condensed_question present; answers are still never logged
```

**Probes are quiet:** successful `/health` and `/ready` calls log at DEBUG, so they don't show at INFO.

```bash
curl -s $BASE/ready >/dev/null; grep -c '"route": "/ready"' /tmp/app.log    # → 0
```

**Readable output:** `LOG_FORMAT=console uvicorn app.main:app --port 8001` prints the same events as coloured key=value lines.

### 2d. Docker: the same ID in nginx and in the API

```bash
export BASE=http://localhost:8000
chat "What is acne?" verify-log-4 -H 'X-Request-ID: verify-nginx-1'
docker compose logs nginx | grep 'request_id=verify-nginx-1'
# → ... "POST /api/chat HTTP/1.1" 200 ... upstream=192.168.97.x:8000 request_id=verify-nginx-1
docker compose logs api --no-log-prefix | jq -c 'select(.request_id=="verify-nginx-1") | .event'

# No X-Request-ID from the client: nginx creates one and the API uses it.
chat "What is acne?" verify-log-4
```

### Expected

- [ ] A generated 32-hex ID, a valid client ID echoed back, an unsafe one replaced.
- [ ] 404/422 bodies and SSE `error` events contain `request_id`, matching the header.
- [ ] One request gives `request_started` → `query_condensed` → `retrieval_done` → `llm_done` → `request_finished`, all with the same `request_id` and `session_id`.
- [ ] `grep` finds no question text and no keys. `LOG_CONTENT=true` adds `question` and `condensed_question`, plus a warning.
- [ ] nginx's access log and the API's JSON lines show the same `request_id`.

---

## 3. Rate limiting with Redis

**What should happen:**
- `/api/chat` and `/api/chat/stream` allow **10 calls per minute per session** and **30 per minute per IP**, counted in Redis and shared by every worker and container.
- Over a limit: **429** with `Retry-After`.
- `POST /api/feedback` has its own per-IP counter (30/min).
- If Redis is unreachable, requests are allowed (fail open).

Run against the Docker stack, since 3 replicas show that the counter is shared:

```bash
export BASE=http://localhost:8000
rcli --scan --pattern 'ratelimit:*' | xargs -r docker compose exec -T redis redis-cli DEL   # start clean
```

### 3a. Per-session limit: 10 per minute

These calls use the LLM, so they take a few seconds each. To make the test faster, ask something out of scope: "How do I bake bread?" is answered without calling the LLM.

```bash
SID=verify-rl-1
for i in $(seq 11); do
  printf "%2d  " $i
  curl -s -o /tmp/verify-body -D /tmp/h $BASE/api/chat -H 'content-type: application/json' \
       -d '{"message":"How do I bake bread?","session_id":"'$SID'"}' -w '%{http_code}'
  printf "  remaining=%s  retry-after=%s  via %s\n" \
    "$(grep -i x-ratelimit-remaining /tmp/h | tr -d '\r' | cut -d' ' -f2)" \
    "$(grep -i retry-after /tmp/h | tr -d '\r' | cut -d' ' -f2)" \
    "$(grep -i x-upstream /tmp/h | tr -d '\r' | cut -d' ' -f2)"
done
jq -c . /tmp/verify-body
# →  1..10: 200, remaining 9 → 0, served by different replicas
# →  11:    429, retry-after = seconds left in this minute
# → {"detail":"Rate limit exceeded. Try again in N seconds.","code":"rate_limited","request_id":"..."}
```

**The counter lives in Redis, shared by all replicas:**

```bash
rcli --scan --pattern 'ratelimit:*'          # ratelimit:session:verify-rl-1:<minute>, ratelimit:ip:<ip>:<minute>
KEY=$(rcli --scan --pattern "ratelimit:session:$SID:*" | tr -d '\r')
rcli GET $KEY                                # → 11
rcli TTL $KEY                                # → ≤ 60, and NOT reset by later hits (EXPIRE ... NX)
docker compose logs api --no-log-prefix | jq -c 'select(.event=="rate_limited") | {limited_by, limit, retry_after}' | tail -1
# → {"limited_by":"session","limit":10,"retry_after":...}   (no IP address in the log)
```

**A new minute is a new window:** wait for the `retry-after` seconds, then:

```bash
chat "How do I bake bread?" $SID            # → 200, X-RateLimit-Remaining: 9
```

### 3b. Per-IP limit: 30 per minute, even when session IDs rotate

```bash
rcli --scan --pattern 'ratelimit:*' | xargs -r docker compose exec -T redis redis-cli DEL
for i in $(seq 31); do
  curl -s -o /dev/null -w '%{http_code} ' $BASE/api/chat -H 'content-type: application/json' \
       -d '{"message":"How do I bake bread?","session_id":"verify-rot-'$i'"}'
done; echo
# → 30 × 200, then 429 (each session used only once: the IP counter caught it)
docker compose logs api --no-log-prefix | jq -c 'select(.event=="rate_limited") | .limited_by' | tail -1   # → "ip"
```

**A faked `X-Forwarded-For` doesn't help behind nginx.** nginx overwrites the header with the real address:

```bash
chat "How do I bake bread?" verify-rot-99 -H 'X-Forwarded-For: 1.2.3.4'       # still 429
rcli --scan --pattern 'ratelimit:ip:*'     # only the real client address, never 1.2.3.4
```

(On a local uvicorn without nginx, `X-Forwarded-For` from 127.0.0.1 **is** trusted. That's uvicorn's default `FORWARDED_ALLOW_IPS`, and it's handy for simulating other IPs. A client connecting from elsewhere can't do that.)

### 3c. What is and isn't limited

```bash
for i in $(seq 5); do curl -s -o /dev/null -D - $BASE/health | grep -ci x-ratelimit; done | sort | uniq -c
# → "5 0": /health, /ready and /history never carry rate-limit headers and never return 429

rcli --scan --pattern 'ratelimit:*' | xargs -r docker compose exec -T redis redis-cli DEL
curl -s -D - -o /dev/null $BASE/api/chat/stream -H 'content-type: application/json' \
     -d '{"message":"How do I bake bread?","session_id":"verify-rl-2"}' | grep -i x-ratelimit
# → the streaming route is limited too: X-RateLimit-Limit / -Remaining present
```

**Feedback has its own counter**, so rating answers never uses up the chat allowance. Feedback needs tracing; with tracing off it returns 503, but it is still counted:

```bash
for i in $(seq 31); do curl -s -o /dev/null -w '%{http_code} ' $BASE/api/feedback \
  -H 'content-type: application/json' -d '{"trace_id":"'$(printf 'a%.0s' {1..32})'","rating":1}'; done; echo
# → 30 × (202, or 503 if tracing is off), then 429
rcli --scan --pattern 'ratelimit:feedback:*'
chat "How do I bake bread?" verify-rl-3      # chat still allowed: separate counter
```

### 3d. Redis down: fail open

```bash
docker compose pause redis
chat "How do I bake bread?" verify-rl-4
docker compose logs api --no-log-prefix --since 1m | jq -c 'select(.event=="rate_limit_unavailable" or .event=="dependency_unavailable") | {event, detail}'
docker compose unpause redis
```

**What you'll see, and why:**
- **The limiter:** it waits at most 0.5 s, logs a single `rate_limit_unavailable` warning (once per 30 s per process), and **allows** the request.
- **The request still gets a 503** (`Chat memory is temporarily unavailable`, after about 6 s), because **chat history uses the same Redis** and can't be read. The 503 has no `X-RateLimit-*` headers.
- **So failing open only makes a difference when Redis is slow,** not completely down. The README ("Rate limiting") explains this trade-off.

### 3e. Turning it off

```bash
# Local: no headers, never 429
MEMORY_BACKEND=redis RATE_LIMIT_ENABLED=false uvicorn app.main:app --port 8001
BASE=http://localhost:8001 chat "How do I bake bread?" verify-rl-5    # no X-RateLimit-* headers
```

### Expected

- [ ] Calls 1–10 in a session return 200 with `Remaining` going from 9 to 0, served by different replicas; call 11 returns **429** with `Retry-After` and `code: rate_limited`.
- [ ] `ratelimit:session:…` reaches 11 in Redis; its TTL is ≤ 60 and isn't reset by later calls.
- [ ] The next minute returns 200 again.
- [ ] 31 rotating session IDs from one IP: the 31st call is 429 with `limited_by: ip`. A faked `X-Forwarded-For` behind nginx changes nothing.
- [ ] `/health` and `/history` are never limited. The stream route is. Feedback has its own counter.
- [ ] With Redis paused: one `rate_limit_unavailable` warning, the limiter doesn't block, and chat history returns 503.

---

## 4. Langfuse tracing and feedback

**What should happen:**
- Each chat request becomes one Langfuse trace: the rewritten question, the chunks with their scores and pages, the exact prompt, and the answer with token counts.
- Phone numbers, emails, Aadhaar and PAN are masked before anything is sent.
- 👍/👎 is stored as a `user_feedback` score on that trace.

### 4a. One-time setup

1. Sign up at <https://cloud.langfuse.com> (**EU** region), create an organisation and a project, then go to **Project Settings → API Keys → Create new API keys**.
2. Add the keys to `.env`. Don't paste them anywhere else:
   ```bash
   cat >> .env <<'EOF'
   TRACING_ENABLED=true
   LANGFUSE_PUBLIC_KEY=pk-lf-...
   LANGFUSE_SECRET_KEY=sk-lf-...
   EOF
   ```
3. Restart so the new settings are read.
   - **Docker:** `docker compose up -d --force-recreate --scale api=3`. Containers read `.env` only when they're created.
   - **Local:** restart uvicorn.
4. Check:
   ```bash
   docker compose logs api --no-log-prefix | jq -c 'select(.event=="tracing_ready") | {enabled, host, environment}'
   # → {"enabled":true,"host":"https://cloud.langfuse.com","environment":"development"}
   ```

### 4b. A trace per answer

```bash
export BASE=http://localhost:8000
SID=verify-lf-1
chat "What is acne?" $SID                    # → body has "trace_id": "<32 hex>"
TID=$(jq -r .trace_id /tmp/verify-body); echo $TID
stream "Who usually gets it?" $SID           # → last line: done: {... "trace_id": "..."}

# The same trace_id is on the log lines, so logs and traces link both ways:
docker compose logs api --no-log-prefix | jq -c 'select(.trace_id=="'$TID'") | .event'
```

Open **Langfuse → Tracing → Traces** and click the newest `chat` trace, or search for `$TID`. Traces can take a few seconds to appear.

**Checklist in the trace view:**

| Where | Expect |
|---|---|
| Tree | `chat` → `condense`, `retrieve`, `generate` (the stream's trace is named `chat-stream`) |
| `chat` | input `{question}`, output `{answer, cited}`, metadata `outcome: ok`, `condense_strategy` |
| `condense` (2nd trace) | input: the chat **transcript** and the follow-up; output `standalone_question` (e.g. "Who usually gets acne?"), `strategy: llm` |
| `retrieve` | input `{query}`; output `chunks` with `id`, `score`, `page`, `source`, `passed`; metadata `attempts`, `k`, `min_score` |
| `generate` | input: the full prompt (system + numbered context + history + question); output: the answer; model `gemma3:4b`, input/output tokens; time to first token on the stream |
| Trace metadata | `request_id`, `prompt_version`, `prompt_sha`, `llm_provider`, `llm_model`, `embedding_model`, `index`, `k`, `min_score` |
| **Sessions** tab | session `verify-lf-1` groups both traces |

**Find the trace for a log line:**

```bash
docker compose logs api --no-log-prefix | jq -r 'select(.event=="request_finished" and .trace_id) | "\(.request_id) -> \(.trace_id)"' | tail -3
```

### 4c. Condense fallback is visible

Run locally with a tiny rewrite timeout. The keys come from `.env`:

```bash
export BASE=http://localhost:8001
MEMORY_BACKEND=redis CONDENSE_TIMEOUT_SECONDS=0.01 uvicorn app.main:app --port 8001 2>&1 | tee /tmp/app.log
chat "What is acne?" verify-lf-2
chat "Who usually gets it?" verify-lf-2
```

**In that second trace:**
- `condense` is **WARNING** with `reason: timeout`;
- `retrieve` has two children, `retrieve.plain` ("Who usually gets it?") and `retrieve.combined` ("What is acne? Who usually gets it?");
- the chunks below the threshold show `passed: false`.

### 4d. Masking

```bash
export BASE=http://localhost:8000
chat "My email is test.user@example.com, phone +91 98765 43210, PAN ABCDE1234F, Aadhaar 2345 6789 0123. What is acne?" verify-lf-3
```

**In that trace (the `chat` input and the `generate` prompt):**
- you see `[EMAIL]`, `[PHONE]`, `[PAN]` and `[AADHAAR]`;
- you **never** see `test.user@example.com`, `98765 43210`, `ABCDE1234F` or `2345 6789 0123`.

Use the trace view's search to make sure the raw values appear nowhere. Doses, pages and years such as "500 mg", "p. 43" or "2023" are left alone.

### 4e. Feedback

**In the browser:** open <http://localhost:8000> and ask a question.
1. Click **👍**. It's highlighted, "Thanks for your feedback!" appears, and a comment box opens.
2. Click **👎** and type a comment such as "too vague, email me at a@b.com", then **Send**.
3. Reload the page. The old answer still has 👍/👎, because restored history keeps `trace_id`.

**In Langfuse:** open that trace and go to **Scores** (or **Tracing → Scores**). Expect **one** `user_feedback` score with value **-1** and the comment `too vague, email me at [EMAIL]`. It replaced the 👍; it wasn't added next to it.

**With curl:**

```bash
curl -s -w ' %{http_code}\n' $BASE/api/feedback -H 'content-type: application/json' \
     -d '{"trace_id":"'$TID'","rating":1,"comment":"accurate"}'
# → {"status":"accepted"} 202   (Langfuse now shows value 1, "accurate")

curl -s -w ' %{http_code}\n' $BASE/api/feedback -H 'content-type: application/json' -d '{"trace_id":"nope","rating":1}'
curl -s -w ' %{http_code}\n' $BASE/api/feedback -H 'content-type: application/json' -d '{"trace_id":"'$TID'","rating":0}'
# → 422 validation_error (trace_id must be 32 hex; rating 1 or -1; comment ≤ 500 chars)
```

### 4f. Errors are traced too

```bash
# LLM down: stop Ollama (quit the menu-bar app, or `pkill ollama`), then:
chat "What is acne?" verify-lf-4             # → 503 dependency_unavailable, with trace_id in logs
# Trace: chat = ERROR "LLMUnavailableError", metadata.outcome = dependency_unavailable; generate = ERROR.
# Start Ollama again afterwards.
```

**Stream cut off by the overall cap.** Run locally with `REQUEST_TIMEOUT_SECONDS=2`, then `stream "Explain acne in detail" verify-lf-5`. In that trace the root has outcome `timeout`, and `generate` is **WARNING `cancelled`**, with the partial output kept.

### 4g. A Langfuse outage doesn't hurt chat

```bash
export BASE=http://localhost:8001
MEMORY_BACKEND=redis LANGFUSE_BASE_URL=http://10.255.255.1 uvicorn app.main:app --port 8001 2>&1 | tee /tmp/app.log
time chat "What is acne?" verify-lf-6        # normal latency, trace_id still returned
sleep 20; grep -c "Failed to export spans" /tmp/app.log      # → ≥ 1: errors only in the logs
```

### 4h. Tracing off

```bash
MEMORY_BACKEND=redis TRACING_ENABLED=false uvicorn app.main:app --port 8001
chat "What is acne?" verify-lf-7             # → "trace_id": null; no 👍/👎 in the UI
curl -s $BASE/api/feedback -H 'content-type: application/json' -d '{"trace_id":"'$(printf 'a%.0s' {1..32})'","rating":1}' | jq -c '{code}'
# → {"code":"tracing_disabled"} (503)
```

### 4i. Retention script (30 days)

```bash
python -m scripts.langfuse_retention                 # dry run: "Found N trace(s) older than ..."; deletes nothing
python -m scripts.langfuse_retention --days 30 --yes # delete traces older than 30 days (permanent)
```

**To see it delete something, use a throwaway test project only.** `--days 0` deletes **every** trace in the project:

```bash
python -m scripts.langfuse_retention --days 0          # dry run first: check the count
python -m scripts.langfuse_retention --days 0 --yes    # then confirm in the UI that the traces are gone
```

**Still to check on the free Hobby plan:**
- whether the API still lists traces older than 30 days (Hobby hides them in the UI). If the 30-day dry run always finds 0 while the UI shows nothing older than 30 days, run the script with `--days 29` at least weekly;
- whether the deleted traces' scores also disappear.

### Expected

- [ ] `tracing_ready` shows `enabled: true`; `/api/chat` and the SSE `done` event return a `trace_id`, which also appears in the logs.
- [ ] Each trace has `chat` → `condense` / `retrieve` / `generate` with the fields above, the metadata is complete, and the session groups the turns.
- [ ] Follow-ups show the transcript and rewrite. A forced fallback shows a WARNING and `retrieve.plain` / `retrieve.combined`.
- [ ] Email, phone, PAN and Aadhaar appear masked and never raw.
- [ ] 👍 then 👎 with a comment leaves **one** `user_feedback` score of -1 with the masked comment; buttons survive a reload; bad bodies give 422.
- [ ] An LLM outage gives an ERROR root; a stream timeout gives a `cancelled` generation.
- [ ] An unreachable Langfuse doesn't slow chat. Tracing off gives `trace_id: null` and feedback 503.
- [ ] The retention dry run counts, and `--yes` deletes.

---

## 5. Cleanup

```bash
# Test sessions and rate-limit counters (only verify-* sessions, never real ones)
rcli --scan --pattern 'medical-rag:chat:verify-*' | xargs -r docker compose exec -T redis redis-cli DEL
rcli --scan --pattern 'ratelimit:*'               | xargs -r docker compose exec -T redis redis-cli DEL
rm -f /tmp/verify-headers /tmp/verify-body /tmp/h /tmp/app*.log

# Stop local uvicorn: Ctrl-C.  Stop the Docker stack (keeps the redis-data volume):
docker compose down
docker compose up -d redis           # if you still want Redis for local runs
```

Traces of the `verify-*` sessions stay in Langfuse. Delete them in the UI (Traces → filter by session → delete), or let the retention script remove them after 30 days.
