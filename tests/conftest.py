import asyncio
import hashlib
import os
import time

# Tests must not depend on the developer's local .env (e.g. MEMORY_BACKEND=redis) or shell.
# Settings are read at import time, so set safe values before importing the app.
os.environ.setdefault("PINECONE_API_KEY", "test-key")
os.environ["LLM_PROVIDER"] = "ollama"
os.environ.pop("MEMORY_BACKEND", None)  # REDIS_URL is kept: it only enables the -m redis tests
os.environ["RATE_LIMIT_ENABLED"] = "false"  # tests/test_rate_limit.py enables it explicitly

import pytest
from fastapi.testclient import TestClient
from langchain_core.documents import Document
from langchain_core.language_models.fake_chat_models import FakeListChatModel

from app.config import Settings

Settings.model_config["env_file"] = None  # never read .env during tests
from app.main import create_app
from app.memory import InMemoryChatMemory
from app.rag import RAGService

DOCS = [
    (Document(page_content="Acne is a common skin disease.", metadata={"source": "Medical_book.pdf", "page": 12.0}), 0.91),
    (Document(page_content="Acne is caused by clogged pores.", metadata={"source": "Medical_book.pdf", "page": 13.0}), 0.85),
    (Document(page_content="Unrelated passage about the heart.", metadata={"source": "Medical_book.pdf"}), 0.40),
]
# Below the default MIN_RELEVANCE_SCORE (0.40): must never reach the prompt or citations.
LOW_DOC = (Document(page_content="Sourdough bread needs a starter.", metadata={"source": "Medical_book.pdf", "page": 99.0}), 0.20)
OFF_TOPIC = [LOW_DOC, (Document(page_content="Football history.", metadata={"source": "Medical_book.pdf"}), 0.15)]


def doc_id(doc: Document) -> str:
    return doc.id or hashlib.sha1(doc.page_content.encode()).hexdigest()[:16]


class StubIndex:
    """Stands in for the Pinecone Index.

    describe_index_stats (used by /ready): succeed, raise `error`, or sleep `delay`.
    query (used by retrieval): returns Pinecone-shaped matches for the store's results;
    `query_errors` are raised one per call first (to script retries), `query_delay` sleeps.
    """

    def __init__(self, error: Exception | None = None, delay: float = 0.0, store=None):
        self.error = error
        self.delay = delay
        self.calls: list[dict] = []
        self.store = store
        self.query_errors: list[BaseException] = []
        self.query_delay = 0.0
        self.query_calls: list[dict] = []

    def describe_index_stats(self, **kwargs):
        self.calls.append(kwargs)
        if self.delay:
            time.sleep(self.delay)  # blocking, like the real SDK
        if self.error:
            raise self.error
        return {"dimension": 384, "total_vector_count": 3}

    def query(self, vector, top_k, include_metadata=True, namespace=None, **kwargs):
        self.query_calls.append({"vector": vector, "top_k": top_k, **kwargs})
        if self.query_delay:
            time.sleep(self.query_delay)
        if self.query_errors:
            raise self.query_errors.pop(0)
        results = self.store.results_for(vector)
        if isinstance(results, BaseException):  # per-query failure (dict value is an exception)
            raise results
        results = results[:top_k]
        return {
            "matches": [
                {"id": doc_id(doc), "score": score, "metadata": {**doc.metadata, "text": doc.page_content}}
                for doc, score in results
            ]
        }


class StubEmbeddings:
    """The "vector" is the query text itself, so the stub index can answer per query."""

    def __init__(self, store):
        self.store = store

    def embed_query(self, text: str):
        self.store.queries.append(text)
        return text


class StubVectorStore:
    """Records queries; returns fixed (Document, score) pairs, or per-query ones from a dict."""

    _text_key = "text"
    _namespace = None

    def __init__(self, results=DOCS + [LOW_DOC]):
        self.results = results
        self.queries: list[str] = []
        self.embeddings = StubEmbeddings(self)
        self.index = StubIndex(store=self)

    def results_for(self, query: str):
        if isinstance(self.results, dict):
            return self.results.get(query, [])
        return self.results


@pytest.fixture
def settings() -> Settings:
    return Settings(pinecone_api_key="test-key", llm_provider="ollama", retriever_k=4, min_relevance_score=0.40)


def make_rag(settings: Settings, responses: list[str], results=None) -> tuple[RAGService, StubVectorStore]:
    store = StubVectorStore() if results is None else StubVectorStore(results)
    rag = RAGService(settings, llm=FakeListChatModel(responses=responses), vectorstore=store)
    rag.retry_backoff_initial = rag.retry_backoff_max = 0  # no real sleeping between retries
    return rag, store


@pytest.fixture
def make_client(settings):
    """Factory: build a TestClient whose RAG LLM replies with the given responses in order."""
    clients = []

    def _make(responses: list[str], results=None, memory=None):
        rag, store = make_rag(settings, responses, results)
        app = create_app()
        app.state.rag = rag
        app.state.memory = memory or InMemoryChatMemory(max_turns=5, ttl_seconds=3600)
        client = TestClient(app)
        client.__enter__()  # run lifespan (it keeps the pre-set fakes)
        clients.append(client)
        return client, store, app.state.memory

    yield _make
    for c in clients:
        c.__exit__(None, None, None)


def run(coro):
    """Await a memory call from a synchronous TestClient test (in-memory backend only)."""
    return asyncio.run(coro)
