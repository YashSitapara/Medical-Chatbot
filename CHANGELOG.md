# Changelog

All notable changes to this project are documented here.
The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project
uses [Semantic Versioning](https://semver.org/). The version lives in `app/__init__.py`; release tags are `v<version>`.

## [0.1.0] - 2026-11-27

First public release of a medical question-answering chatbot built with RAG
(Retrieval-Augmented Generation). It answers questions from medical documents
with inline citations, streams responses, remembers conversations, and includes
the observability and resilience features needed to run it in production.

> ⚠️ For learning purposes only. Not medical advice.

## ✨ Features

### Question answering
- RAG pipeline: HuggingFace embeddings (`all-MiniLM-L6-v2`) + Pinecone vector search
- Inline citations [1][2] with source file, page number and snippet
- Relevance threshold: off-topic questions get "not found" without calling the LLM
- Follow-up questions rewritten into standalone queries using chat history
- Fallback when the rewrite fails: merged search over the original and context-enriched query
- Runs locally on Ollama `gemma3:4b` by default; OpenAI (`gpt-4o-mini`) is optional via `LLM_PROVIDER=openai`

### API & UI
- FastAPI backend with typed request/response models and auto-generated docs at `/docs`
- Streaming answers over Server-Sent Events (`session`, `citations`, `token`, `done`, `error`)
- Simple web chat UI with 👍/👎 feedback on each answer

### Memory
- Redis-backed chat history: works across multiple workers and survives restarts
- Per-session TTL and history trimming; in-memory backend available for local dev

### Production readiness
- `/health` (liveness) and `/ready` (checks Redis and Pinecone)
- Structured JSON logs with request IDs; no user content in logs by default
- Timeouts, retries with exponential backoff and jitter, and clear 503/504/500 error mapping
- Redis-based rate limiting (429 + `Retry-After`), fail-open if Redis is unavailable
- Langfuse tracing of every request (condense → retrieve → generate), with PII masking,
  config metadata and user feedback linked to traces
- Horizontal scaling: nginx load-balances multiple API replicas (`docker compose up --scale api=3`);
  chat history and rate limits are shared through Redis, so any replica can serve any request

### Security
- Session IDs are server-issued random UUID4s and act as bearer tokens; the API rejects any
  other value (422), so a client can't choose a guessable ID that someone else could read
- CORS is closed by default; cross-origin browser access needs `CORS_ORIGINS` set explicitly

### Dev & deployment
- Docker Compose setup (nginx + API + Redis): one command to run everything
- Test suite with fakes and fakeredis; runs offline without API keys (CI not set up yet)

## 🚀 Quick start
See the [README](README.md#quick-start-local).

## ⚠️ Known limitations
- No formal evaluation yet. Retrieval and answer quality are not measured (planned for v0.2.0)
- Fixed-window rate limiting allows short bursts at window boundaries
- Not yet deployed to the cloud
- No CI pipeline yet
- Page numbers in citations require re-running ingest (`python -m scripts.ingest --recreate`) on indexes built before page metadata was stored

[0.1.0]: https://github.com/YashSitapara/Medical-Chatbot/releases/tag/v0.1.0
