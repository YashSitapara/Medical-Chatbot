from types import SimpleNamespace

import pytest
from langchain_core.documents import Document

from app.config import Settings
from app.rag import build_citations
from scripts.ingest import chunk_ids, ensure_index, normalise_metadata, split


def raw_page(page, text="Acne is a common skin disease. " * 40):
    # Shape of what PyPDFLoader returns: 0-based page index plus extra keys.
    meta = {"source": "data/Medical_book.pdf", "page_label": str(page + 1) if page is not None else "", "total_pages": 637}
    if page is not None:
        meta["page"] = page
    return Document(page_content=text, metadata=meta)


def test_normalise_converts_zero_based_page_to_one_based():
    doc = normalise_metadata([raw_page(42)])[0]
    assert doc.metadata == {"source": "Medical_book.pdf", "page": 43}


def test_normalise_first_page_and_missing_page():
    assert normalise_metadata([raw_page(0)])[0].metadata["page"] == 1
    assert normalise_metadata([raw_page(None)])[0].metadata["page"] is None


def test_split_keeps_page_and_drops_none():
    chunks = split(normalise_metadata([raw_page(42), raw_page(None)]), chunk_size=200, chunk_overlap=20)
    with_page = [c for c in chunks if "page" in c.metadata]
    without = [c for c in chunks if "page" not in c.metadata]
    assert with_page and without  # both pages produced chunks
    assert all(c.metadata["page"] == 43 for c in with_page)
    assert all(None not in c.metadata.values() for c in chunks)


def test_chunk_ids_deterministic_and_unique():
    chunks = split(normalise_metadata([raw_page(42), raw_page(43)]), chunk_size=200, chunk_overlap=20)
    ids = chunk_ids(chunks)
    assert ids == chunk_ids(chunks)
    assert len(set(ids)) == len(ids)


def test_page_number_end_to_end_ingest_to_citation():
    chunk = split(normalise_metadata([raw_page(42)]), 500, 50)[0]
    # Pinecone returns numeric metadata as floats.
    stored = Document(page_content=chunk.page_content, metadata={**chunk.metadata, "page": float(chunk.metadata["page"])})
    assert build_citations([(stored, 0.9)])[0].page == 43  # PDF index 42 == printed page 43


class FakePinecone:
    def __init__(self, existing_dim=None):
        self.dim = existing_dim
        self.created = None
        self.deleted = False

    def has_index(self, name):
        return self.dim is not None

    def delete_index(self, name):
        self.deleted, self.dim = True, None

    def create_index(self, name, dimension, metric, spec):
        self.created = {"dimension": dimension, "metric": metric}
        self.dim = dimension

    def describe_index(self, name):
        return SimpleNamespace(dimension=self.dim, status={"ready": True})


@pytest.fixture
def settings():
    return Settings(pinecone_api_key="k")


def test_ensure_index_creates_with_model_dimension(settings):
    pc = FakePinecone()
    ensure_index(pc, settings, dim=384, recreate=False)
    assert pc.created == {"dimension": 384, "metric": "cosine"}


def test_ensure_index_refuses_mismatched_existing_index(settings):
    pc = FakePinecone(existing_dim=768)
    with pytest.raises(RuntimeError, match="--recreate"):
        ensure_index(pc, settings, dim=384, recreate=False)
    assert pc.created is None


def test_ensure_index_recreate_rebuilds_with_new_dimension(settings):
    pc = FakePinecone(existing_dim=768)
    ensure_index(pc, settings, dim=384, recreate=True)
    assert pc.deleted and pc.created["dimension"] == 384
