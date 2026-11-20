"""Single source of truth for the embedding model.

Both the API (query time) and scripts/ingest.py (index time) build embeddings here.
Query vectors and stored vectors are only comparable when they come from the same
model with the same settings, so nothing else should construct an embedder.
"""

from langchain_core.embeddings import Embeddings

from app.config import Settings


def build_embeddings(settings: Settings) -> Embeddings:
    from langchain_huggingface import HuggingFaceEmbeddings

    return HuggingFaceEmbeddings(
        model_name=settings.embedding_model,
        model_kwargs={"device": "cpu"},
        # Unit-length vectors: cosine similarity in Pinecone then equals the dot product,
        # and scores are comparable across queries (used by MIN_RELEVANCE_SCORE).
        encode_kwargs={"normalize_embeddings": True},
    )


def embedding_dimension(embeddings: Embeddings) -> int:
    """Measure the vector size from the model itself instead of trusting a config value."""
    return len(embeddings.embed_query("dimension probe"))


def check_index_dimension(index_dim: int, model_dim: int, model_name: str, index_name: str) -> None:
    if index_dim != model_dim:
        raise RuntimeError(
            f"Embedding dimension mismatch: Pinecone index '{index_name}' stores {index_dim}-d vectors, "
            f"but EMBEDDING_MODEL '{model_name}' produces {model_dim}-d vectors. "
            "Use the model the index was built with, or rebuild it with "
            "`python -m scripts.ingest --recreate`."
        )
