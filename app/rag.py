"""Retrieval-augmented generation: query condensing, retrieval, citations, streaming."""

import asyncio
import logging
import re
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any, Literal

from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.vectorstores import VectorStore

from app.config import Settings
from app.embeddings import build_embeddings, check_index_dimension, embedding_dimension
from app.memory import to_langchain
from app.schemas import Citation, Message

logger = logging.getLogger(__name__)

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


def clean_rewrite(text: str, question: str) -> str | None:
    """Return the rewritten question, or None if the model's output looks like an answer."""
    line = next((ln.strip() for ln in text.strip().splitlines() if ln.strip()), "")
    line = _REWRITE_LABEL_RE.sub("", line).strip().strip("\"'“”‘’`").strip()
    if not line:
        return None
    if re.search(r"\[\d", line):  # only answers cite passages
        return None
    if len(line) > max(200, 4 * len(question)):
        return None
    if not line.endswith("?"):  # an answer is a statement
        return None
    if len([s for s in re.split(r"(?<=[.!?])\s+", line) if s]) > 1:
        return None
    return line


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
        )

    from langchain_ollama import ChatOllama

    return ChatOllama(
        model=settings.ollama_model,
        base_url=settings.ollama_base_url,
        temperature=settings.llm_temperature,
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
    logger.info("Embedding model %s (%d-d) matches index %s", settings.embedding_model, model_dim, settings.pinecone_index_name)
    return store


# ---------------------------------------------------------------------------
# Service
# ---------------------------------------------------------------------------


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

    def ping_vectorstore(self, timeout: float | None = None) -> None:
        """Cheapest Pinecone round trip: index stats (no embedding, no query). Blocking.

        `timeout` is passed to the HTTP request itself, so a hung call ends on its own
        even after the async caller has stopped waiting for this thread.
        """
        kwargs = {"_request_timeout": timeout} if timeout else {}
        self.vectorstore.index.describe_index_stats(**kwargs)

    async def condense_question(self, question: str, history: list[Message]) -> str:
        """Turn a follow-up into a standalone search query. No-op without history."""
        if not history:
            return question
        raw = await self._condense_chain.ainvoke({"transcript": format_transcript(history), "question": question})
        standalone = clean_rewrite(raw, question)
        if standalone:
            logger.info("Condensed %r -> %r", question, standalone)
            return standalone
        # The model answered (or rambled) instead of rewriting. Search with the previous user
        # question + the follow-up: its embedding still carries the topic.
        previous = next((m.content for m in reversed(history) if m.role == "user"), "")
        fallback = f"{previous} {question}".strip()
        logger.warning("Rejected rewrite %r for %r; searching with %r", raw.strip()[:200], question, fallback)
        return fallback

    async def retrieve(self, query: str) -> list[tuple[Document, float]]:
        """Top-k search, keeping only chunks at or above the relevance threshold."""
        # Embedding + Pinecone query are blocking calls; keep them off the event loop.
        results = await asyncio.to_thread(self.vectorstore.similarity_search_with_score, query, k=self.k)
        kept = [(doc, score) for doc, score in results if score >= self.min_score]
        logger.info(
            "Retrieved for %r: kept %d/%d (min_score=%.2f, top=%s)",
            query,
            len(kept),
            len(results),
            self.min_score,
            f"{results[0][1]:.3f}" if results else "n/a",
        )
        return kept

    async def _prepare(self, question: str, history: list[Message]):
        """Returns (relevant results, prompt) — prompt is None when nothing is relevant."""
        query = await self.condense_question(question, history)
        results = await self.retrieve(query)
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
            return NO_CONTEXT_ANSWER, []
        message = await self.llm.ainvoke(prompt_value)
        answer = normalize_citations(StrOutputParser().invoke(message).strip(), len(results))
        return answer, build_citations(results, answer)

    async def stream(self, question: str, history: list[Message]) -> AsyncIterator[StreamEvent]:
        results, prompt_value = await self._prepare(question, history)
        yield StreamEvent("citations", build_citations(results))

        if prompt_value is None:  # same event shape as a normal answer, so clients need no special case
            yield StreamEvent("token", NO_CONTEXT_ANSWER)
            yield StreamEvent("done", {"answer": NO_CONTEXT_ANSWER, "citations": []})
            return

        parts: list[str] = []
        async for chunk in self.llm.astream(prompt_value):
            text = chunk.content if isinstance(chunk.content, str) else ""
            if text:
                parts.append(text)
                yield StreamEvent("token", text)

        # Tokens were streamed raw; the final (stored, re-rendered) text uses the [1][3] form.
        answer = normalize_citations("".join(parts).strip(), len(results))
        yield StreamEvent("done", {"answer": answer, "citations": build_citations(results, answer)})
