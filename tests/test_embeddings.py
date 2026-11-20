import pytest
from langchain_core.embeddings import DeterministicFakeEmbedding

from app.embeddings import check_index_dimension, embedding_dimension


def test_embedding_dimension_is_measured_from_model():
    assert embedding_dimension(DeterministicFakeEmbedding(size=384)) == 384
    assert embedding_dimension(DeterministicFakeEmbedding(size=768)) == 768


def test_matching_dimension_passes():
    check_index_dimension(384, 384, "all-MiniLM-L6-v2", "medical-chatbot")


def test_mismatched_dimension_raises_with_reingest_hint():
    with pytest.raises(RuntimeError, match=r"stores 384-d.*produces 768-d.*ingest --recreate"):
        check_index_dimension(384, 768, "all-mpnet-base-v2", "medical-chatbot")


def test_api_and_ingest_share_one_embedding_factory():
    import app.rag as rag
    import scripts.ingest as ingest
    from app import embeddings

    assert rag.build_embeddings is embeddings.build_embeddings
    assert ingest.build_embeddings is embeddings.build_embeddings
