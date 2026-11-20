"""Retrieval-augmented generation: query condensing, retrieval, citations, streaming."""

import asyncio
import re
import time
from collections.abc import AsyncIterator, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Literal

from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.vectorstores import VectorStore
import structlog
from tenacity import AsyncRetrying, retry_if_exception, stop_after_attempt, wait_exponential_jitter

from app.config import Settings
from app.embeddings import build_embeddings, check_index_dimension, embedding_dimension
from app.errors import (
    DependencyUnavailableError,
    LLMUnavailableError,
    RetrievalUnavailableError,
    is_transient_llm_error,
    is_transient_pinecone_error,
)
from app.memory import to_langchain
from app.request_context import mark_sensitive, set_outcome
from app.schemas import Citation, Message

logger = structlog.get_logger(__name__)


def _ms_since(start: float) -> float:
    return round((time.perf_counter() - start) * 1000, 1)

SNIPPET_CHARS = 300
NO_CONTEXT_ANSWER = (
    "I couldn't find information about this in the medical reference I have access to. "
    "Please consult a healthcare professional."
)
_CITATION_RE = re.compile(r"\[(\d+)\]")
# Any citation-like marker, including grouped forms: [1], [1, 3], [1;3], [1-3], [1–3].
_ANY_CITATION_RE = re.compile(r"\s*\[\d+(?:\s*[,;\-–]\s*\d+)*\]")
# Grouped forms only (a list or a range) - what normalize_citations rewrites.
_GROUPED_CITATION_RE = re.compile(r"\[(\d+(?:\s*[,;]\s*\d+)+|\d+\s*[-–]\s*\d+)\]")
_REWRITE_LABEL_RE = re.compile(r"^(standalone|rewritten)?\s*question\s*:\s*", re.IGNORECASE)

SYSTEM_PROMPT = (
    "You are a medical assistant for question-answering tasks. "
    "Answer ONLY from the numbered context passages below. "
    "Cite every claim inline with the passage number in square brackets, e.g. [1] or [2][3]. "
    "Never group numbers inside one bracket like [1, 3]; write [1][3]. "
    "If the context does not contain the answer, say you don't know - do not make anything up. "
    "Keep the answer concise (at most five sentences). "
    "You are not a substitute for a medical professional; for diagnosis or treatment decisions, "
    "advise the user to consult a doctor.\n\n"
    "Context:\n{context}"
)

# The history goes in as a plain TRANSCRIPT inside one human message, not as chat turns.
# Given real chat turns, a small model sees "my turn to reply" and answers the follow-up
# instead of rewriting it (observed with gemma3:4b after ~3 turns).
CONDENSE_PROMPT = (
    "You rewrite follow-up questions into standalone search queries. You never answer them. "
    "Replace pronouns such as 'it' or 'they' with what they refer to in the chat history. "
    "Output only the rewritten question, on one line, ending with '?'."
)
CONDENSE_HUMAN = (
    "Chat history:\n{transcript}\n\n"
    "Follow-up question: {question}\n"
    "Standalone question:"
)

answer_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", SYSTEM_PROMPT),
        MessagesPlaceholder("history"),
        ("human", "{question}"),
    ]
)

condense_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", CONDENSE_PROMPT),
        ("human", CONDENSE_HUMAN),
    ]
)


@dataclass
class StreamEvent:
    type: Literal["citations", "token", "done"]
    data: Any


# ---------------------------------------------------------------------------
# Pure helpers (unit-tested directly)
# ---------------------------------------------------------------------------


def _page(doc: Document) -> int | None:
    """Return the stored page number as-is.

    The stored value is ALREADY 1-based: PyPDF's 0-based index is converted once, at
    ingest time (scripts/ingest.py: normalise_metadata). Do not add 1 here again.
    """
    page = doc.metadata.get("page")
    # Pinecone stores numbers as floats (43.0); normalise to int.
    return int(page) if isinstance(page, (int, float)) else None


def _source(doc: Document) -> str:
    return str(doc.metadata.get("source", "unknown"))


def format_context(docs: list[Document]) -> str:
    """Number each chunk so the model can cite it as [n]."""
    blocks = []
    for i, doc in enumerate(docs, start=1):
        page = _page(doc)
        location = f"{_source(doc)}, p. {page}" if page is not None else _source(doc)
        blocks.append(f"[{i}] ({location})\n{doc.page_content.strip()}")
    return "\n\n".join(blocks)


def cited_ids(answer: str) -> set[int]:
    """Ids cited as [n]. Run normalize_citations first so grouped forms are counted."""
    return {int(n) for n in _CITATION_RE.findall(answer)}


def normalize_citations(text: str, max_id: int) -> str:
    """Rewrite grouped citations into the single-number form: [1, 3] -> [1][3], [1-3] -> [1][2][3].

    Only rewrites when every number is a valid passage id (1..max_id), so things like
    years ("[2023]") or unrelated bracketed lists are left untouched.
    """

    def expand(match: re.Match) -> str:
        body = match.group(1)
        if re.search(r"[-–]", body):
            start, end = (int(n) for n in re.split(r"\s*[-–]\s*", body))
            ids = list(range(start, end + 1)) if start < end else []
        else:
            ids = [int(n) for n in re.split(r"\s*[,;]\s*", body)]
        if not ids or any(i < 1 or i > max_id for i in ids):
            return match.group(0)
        return "".join(f"[{i}]" for i in ids)

    return _GROUPED_CITATION_RE.sub(expand, text)


def format_transcript(history: list[Message], max_messages: int = 6, max_chars: int = 300) -> str:
    """Recent history as plain text for the rewrite prompt.

    Only the last few turns matter for resolving 'it'; answers are truncated (the rewrite
    needs the topic, not the details) and citation markers are removed so the model is not
    nudged into writing an answer.
    """
    lines = []
    for m in history[-max_messages:]:
        text = " ".join(_ANY_CITATION_RE.sub("", m.content).split())
        if len(text) > max_chars:
            text = text[:max_chars].rstrip() + "…"
        lines.append(f"{'User' if m.role == 'user' else 'Assistant'}: {text}")
    return "\n".join(lines)


# A rewrite may be up to 3x the question, but never rejected below this many characters:
# "Who gets it?" (12 chars) -> "Who usually gets acne vulgaris?" (31) is a good rewrite.
REWRITE_MIN_MAX_CHARS = 150
REWRITE_ANSWER_WORDS = 30
PREVIOUS_QUESTION_MAX_CHARS = 300  # MiniLM reads ~256 tokens; keep room for the follow-up


def check_rewrite(text: str, question: str) -> tuple[str, str | None]:
    """Return (cleaned first line, rejection reason or None if the rewrite is usable).

    Rejected when: empty; longer than max(3x the question, 150 chars); it looks like an
    answer (no '?' and more than 30 words); or it cites a passage like [1]. The citation
    rule catches short answers: the real failure "Teenagers are more likely to develop
    acne... [1]." is only ~20 words.
    """
    line = next((ln.strip() for ln in text.strip().splitlines() if ln.strip()), "")
    line = _REWRITE_LABEL_RE.sub("", line).strip().strip("\"'“”‘’`").strip()
    if not line:
        return line, "empty"
    if re.search(r"\[\d", line):  # only answers cite passages
        return line, "has_citation"
    if len(line) > max(3 * len(question), REWRITE_MIN_MAX_CHARS):
        return line, "too_long"
    if not line.endswith("?") and len(line.split()) > REWRITE_ANSWER_WORDS:
        return line, "looks_like_answer"
    return line, None


def clean_rewrite(text: str, question: str) -> str | None:
    """Return the rewritten question, or None if the model's output looks like an answer."""
    line, reason = check_rewrite(text, question)
    return None if reason else line


def build_citations(results: list[tuple[Document, float | None]], answer: str = "") -> list[Citation]:
    used = cited_ids(answer)
    citations = []
    for i, (doc, score) in enumerate(results, start=1):
        text = " ".join(doc.page_content.split())
        snippet = text if len(text) <= SNIPPET_CHARS else text[:SNIPPET_CHARS].rstrip() + "…"
        citations.append(
            Citation(
                id=i,
                source=_source(doc),
                page=_page(doc),
                snippet=snippet,
                score=round(float(score), 4) if score is not None else None,
                cited=i in used,
            )
        )
    return citations


# ---------------------------------------------------------------------------
# Factories for the real dependencies
# ---------------------------------------------------------------------------


def build_llm(settings: Settings) -> BaseChatModel:
    if settings.llm_provider == "openai":
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            model=settings.openai_model,
            temperature=settings.llm_temperature,
            api_key=settings.openai_api_key,
            stream_usage=True,  # token counts for streamed answers too (llm_done)
            timeout=settings.llm_timeout_seconds,
            # Retries happen only before the first token: streamed text can't be taken back.
            max_retries=settings.llm_max_retries,
        )

    from langchain_ollama import ChatOllama

    return ChatOllama(
        model=settings.ollama_model,
        base_url=settings.ollama_base_url,
        temperature=settings.llm_temperature,
        # httpx timeout for both the sync and async clients. Ollama is local: no retries.
        client_kwargs={"timeout": settings.llm_timeout_seconds},
    )


def build_vectorstore(settings: Settings) -> VectorStore:
    from langchain_pinecone import PineconeVectorStore

    embeddings = build_embeddings(settings)
    store = PineconeVectorStore(
        index_name=settings.pinecone_index_name,
        embedding=embeddings,
        pinecone_api_key=settings.pinecone_api_key.get_secret_value(),
    )
    # Fail at startup, not on the first question, if the index was built with another model.
    index_dim = store.index.describe_index_stats().dimension
    model_dim = embedding_dimension(embeddings)
    check_index_dimension(index_dim, model_dim, settings.embedding_model, settings.pinecone_index_name)
    logger.info("embedding_index_checked", model=settings.embedding_model, dim=model_dim, index=settings.pinecone_index_name)
    return store


# ---------------------------------------------------------------------------
# Service
# ---------------------------------------------------------------------------


@dataclass
class RetrievalPlan:
    """How the search query was chosen. Kept for tracing (condense_strategy)."""

    strategy: Literal["skipped", "llm", "fallback_dual"]
    query: str  # the search query (fallback_dual: the plain follow-up)
    previous_question: str | None = None  # fallback_dual only
    reason: str | None = None  # why the rewrite was rejected (fallback_dual only)


@contextmanager
def _llm_errors(stage: str) -> Iterator[None]:
    """Turn provider timeouts / connection errors / 5xx into LLMUnavailableError (503).

    Anything else (e.g. 401 from a bad key, or a bug) propagates and becomes a 500.
    """
    try:
        yield
    except DependencyUnavailableError:
        raise
    except Exception as exc:
        if is_transient_llm_error(exc):
            raise LLMUnavailableError(f"LLM {stage} failed: {type(exc).__name__}") from exc
        raise


class RAGService:
    def __init__(
        self,
        settings: Settings,
        llm: BaseChatModel | None = None,
        vectorstore: VectorStore | None = None,
    ) -> None:
        self.settings = settings
        self.k = settings.retriever_k
        self.min_score = settings.min_relevance_score
        self.llm = llm or build_llm(settings)
        self.vectorstore = vectorstore or build_vectorstore(settings)
        self._condense_chain = condense_prompt | self.llm | StrOutputParser()
        # tenacity backoff between Pinecone attempts (seconds); tests set these to 0.
        self.retry_backoff_initial = 0.2
        self.retry_backoff_max = 2.0

    def ping_vectorstore(self, timeout: float | None = None) -> None:
        """Cheapest Pinecone round trip: index stats (no embedding, no query). Blocking.

        `timeout` is passed to the HTTP request itself, so a hung call ends on its own
        even after the async caller has stopped waiting for this thread.
        """
        kwargs = {"_request_timeout": timeout} if timeout else {}
        self.vectorstore.index.describe_index_stats(**kwargs)

    async def plan_retrieval(self, question: str, history: list[Message]) -> RetrievalPlan:
        """Decide the search query. Never fails: a failed/timed-out/bad rewrite falls back.

        Logs `query_condensed` (metadata only; the text itself only with LOG_CONTENT).
        """
        start = time.perf_counter()
        if not history:
            plan = RetrievalPlan("skipped", question)
            self._log_condensed(question, plan, history, start)
            return plan

        raw: str | None = None
        try:
            raw = await asyncio.wait_for(
                self._condense_chain.ainvoke({"transcript": format_transcript(history), "question": question}),
                self.settings.condense_timeout_seconds,
            )
            standalone, reason = check_rewrite(raw, question)
        except TimeoutError:
            reason = "timeout"
        except Exception as exc:  # the request must continue without a rewrite
            reason = "call_failed"
            logger.warning("condense_failed", error=type(exc).__name__)

        if reason is None:
            mark_sensitive(standalone)
            plan = RetrievalPlan("llm", standalone)
        else:
            previous = next((m.content for m in reversed(history) if m.role == "user"), "")
            previous = previous[:PREVIOUS_QUESTION_MAX_CHARS]
            mark_sensitive(previous, raw)
            extra = {"rewrite_raw": raw.strip()[:200]} if (raw and self.settings.log_content) else {}
            logger.warning("rewrite_rejected", reason=reason, **extra)
            plan = RetrievalPlan("fallback_dual", question, previous_question=previous or None, reason=reason)
        self._log_condensed(question, plan, history, start)
        return plan

    def _log_condensed(self, question: str, plan: RetrievalPlan, history: list[Message], start: float) -> None:
        extra = {"condensed_question": plan.query} if self.settings.log_content else {}
        if plan.reason:
            extra["reason"] = plan.reason
        logger.info(
            "query_condensed",
            condense_strategy=plan.strategy,  # skipped (no history) | llm | fallback_dual
            changed=plan.query.strip() != question.strip(),
            history_messages=len(history),
            latency_ms=_ms_since(start),
            **extra,
        )

    async def retrieve(self, query: str, kind: str = "standalone") -> list[tuple[Document, float]]:
        """Embed locally, query Pinecone (timeout + retries), keep chunks above the threshold."""
        start = time.perf_counter()
        # Local CPU work, deterministic: no timeout/retry needed. Off the event loop.
        vector = await asyncio.to_thread(self.vectorstore.embeddings.embed_query, query)
        response = await self._query_index(vector)
        results = self._to_documents(response)
        kept = [(doc, score) for doc, score in results if score >= self.min_score]
        scores = [round(float(score), 3) for _, score in results]  # before filtering: shows what was cut
        logger.info(
            "retrieval_done",
            query_kind=kind,  # standalone | plain | combined (the last two: condense fallback)
            chunks_retrieved=len(results),
            chunks_passed=len(kept),
            min_score=self.min_score,
            top_score=scores[0] if scores else None,
            scores=scores,
            latency_ms=_ms_since(start),
        )
        return kept

    async def _query_index(self, vector: list[float]):
        """One Pinecone query: per-attempt timeout, retried on timeouts/connection errors/5xx.

        Not PineconeVectorStore.similarity_search_with_score: it can't pass a request timeout,
        so a timed-out attempt would leave its thread running. Here the HTTP call itself is
        limited (`_request_timeout`) and asyncio stops waiting at the same moment.
        """
        timeout = self.settings.retrieval_timeout_seconds
        store = self.vectorstore

        def query():
            return store.index.query(
                vector=vector,
                top_k=self.k,
                include_metadata=True,
                namespace=getattr(store, "_namespace", None),
                _request_timeout=timeout,
            )

        def log_retry(state) -> None:
            logger.warning(
                "retrieval_retry",
                attempt=state.attempt_number,
                error=type(state.outcome.exception()).__name__,
            )

        try:
            async for attempt in AsyncRetrying(
                stop=stop_after_attempt(self.settings.retrieval_max_attempts),
                wait=wait_exponential_jitter(multiplier=self.retry_backoff_initial, max=self.retry_backoff_max),
                retry=retry_if_exception(is_transient_pinecone_error),
                before_sleep=log_retry,
                reraise=True,
            ):
                with attempt:
                    return await asyncio.wait_for(asyncio.to_thread(query), timeout)
        except Exception as exc:
            if is_transient_pinecone_error(exc):  # retries exhausted -> 503
                raise RetrievalUnavailableError(f"Pinecone query failed: {type(exc).__name__}") from exc
            raise  # 4xx (bad key, missing index) or a bug -> 500

    def _to_documents(self, response) -> list[tuple[Document, float]]:
        """Pinecone matches -> (Document, score), like langchain-pinecone, plus the chunk id."""
        text_key = getattr(self.vectorstore, "_text_key", "text")
        results = []
        for match in response["matches"]:
            metadata = dict(match.get("metadata") or {})
            if text_key not in metadata:
                logger.warning("match_without_text", text_key=text_key)
                continue
            text = metadata.pop(text_key)
            results.append((Document(id=match.get("id"), page_content=text, metadata=metadata), match["score"]))
        return results

    async def _fallback_retrieve(self, question: str, previous_question: str | None) -> list[tuple[Document, float]]:
        """Search the plain follow-up AND previous question + follow-up; keep each chunk's best score.

        A real follow-up ("who gets it?") scores higher through the combined query; a topic
        switch ("what causes migraines?") through the plain one. No LLM call, both in parallel.
        If one search fails, the other's results are still used.
        """
        queries = [("plain", question)]
        if previous_question:
            queries.append(("combined", f"{previous_question} {question}"))
        outcomes = await asyncio.gather(*(self.retrieve(q, kind=k) for k, q in queries), return_exceptions=True)
        ok = [r for r in outcomes if not isinstance(r, BaseException)]
        if not ok:
            raise next(r for r in outcomes if isinstance(r, BaseException))
        best: dict[str, tuple[Document, float]] = {}
        for results in ok:
            for doc, score in results:
                key = doc.id or doc.page_content[:100]
                if key not in best or score > best[key][1]:
                    best[key] = (doc, score)
        return sorted(best.values(), key=lambda x: x[1], reverse=True)[: self.k]

    def _log_llm(self, start: float, usage: dict | None, ttft_ms: float | None = None) -> None:
        extra = {"ttft_ms": ttft_ms} if ttft_ms is not None else {}
        logger.info(
            "llm_done",
            provider=self.settings.llm_provider,
            model=self.settings.llm_model_name,
            input_tokens=(usage or {}).get("input_tokens"),  # None if the provider doesn't report it
            output_tokens=(usage or {}).get("output_tokens"),
            latency_ms=_ms_since(start),
            **extra,
        )

    async def _prepare(self, question: str, history: list[Message]):
        """Returns (relevant results, prompt) — prompt is None when nothing is relevant."""
        plan = await self.plan_retrieval(question, history)
        if plan.strategy == "fallback_dual":
            results = await self._fallback_retrieve(plan.query, plan.previous_question)
        else:
            results = await self.retrieve(plan.query)
        if not results:
            return results, None
        prompt_value = await answer_prompt.ainvoke(
            {
                "context": format_context([doc for doc, _ in results]),
                "history": to_langchain(history),
                "question": question,
            }
        )
        return results, prompt_value

    async def answer(self, question: str, history: list[Message]) -> tuple[str, list[Citation]]:
        results, prompt_value = await self._prepare(question, history)
        if prompt_value is None:  # out of scope: answer deterministically, skip the LLM
            set_outcome("no_context")
            return NO_CONTEXT_ANSWER, []
        start = time.perf_counter()
        with _llm_errors("answer"):
            message = await self.llm.ainvoke(prompt_value)
        self._log_llm(start, getattr(message, "usage_metadata", None))
        answer = normalize_citations(StrOutputParser().invoke(message).strip(), len(results))
        return answer, build_citations(results, answer)

    async def stream(self, question: str, history: list[Message]) -> AsyncIterator[StreamEvent]:
        results, prompt_value = await self._prepare(question, history)
        yield StreamEvent("citations", build_citations(results))

        if prompt_value is None:  # same event shape as a normal answer, so clients need no special case
            set_outcome("no_context")
            yield StreamEvent("token", NO_CONTEXT_ANSWER)
            yield StreamEvent("done", {"answer": NO_CONTEXT_ANSWER, "citations": []})
            return

        parts: list[str] = []
        start = time.perf_counter()
        ttft_ms = None
        merged = None  # chunks add up; token usage arrives on the last one
        with _llm_errors("stream"):
            async for chunk in self.llm.astream(prompt_value):
                merged = chunk if merged is None else merged + chunk
                text = chunk.content if isinstance(chunk.content, str) else ""
                if text:
                    if ttft_ms is None:
                        ttft_ms = _ms_since(start)
                    parts.append(text)
                    yield StreamEvent("token", text)
        self._log_llm(start, getattr(merged, "usage_metadata", None), ttft_ms=ttft_ms)

        # Tokens were streamed raw; the final (stored, re-rendered) text uses the [1][3] form.
        answer = normalize_citations("".join(parts).strip(), len(results))
        yield StreamEvent("done", {"answer": answer, "citations": build_citations(results, answer)})
