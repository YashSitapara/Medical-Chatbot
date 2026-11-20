"""Load PDFs, chunk them and upsert embeddings into Pinecone.

Usage:
    python -m scripts.ingest                 # upsert (idempotent)
    python -m scripts.ingest --recreate      # drop and rebuild the index
"""

import argparse
import hashlib
import logging
import time
from pathlib import Path

from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pinecone import Pinecone, ServerlessSpec

from app.config import Settings, get_settings
from app.embeddings import build_embeddings, check_index_dimension, embedding_dimension

logger = logging.getLogger("ingest")


def load_pdfs(data_dir: Path) -> list[Document]:
    loader = DirectoryLoader(str(data_dir), glob="**/*.pdf", loader_cls=PyPDFLoader, show_progress=True)
    return loader.load()


def normalise_metadata(docs: list[Document]) -> list[Document]:
    """Keep only what citations need: file name and 1-based page number."""
    out = []
    for doc in docs:
        page = doc.metadata.get("page")
        out.append(
            Document(
                page_content=doc.page_content,
                metadata={
                    "source": Path(doc.metadata.get("source", "unknown")).name,
                    "page": int(page) + 1 if page is not None else None,
                },
            )
        )
    return out


def split(docs: list[Document], chunk_size: int, chunk_overlap: int) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = [c for c in splitter.split_documents(docs) if c.page_content.strip()]
    for c in chunks:  # Pinecone rejects null metadata values
        c.metadata = {k: v for k, v in c.metadata.items() if v is not None}
    return chunks


def chunk_ids(chunks: list[Document]) -> list[str]:
    """Deterministic ids so re-running ingest overwrites instead of duplicating."""
    counters: dict[tuple, int] = {}
    ids = []
    for c in chunks:
        key = (c.metadata.get("source"), c.metadata.get("page"))
        idx = counters.get(key, 0)
        counters[key] = idx + 1
        ids.append(hashlib.sha1(f"{key[0]}:{key[1]}:{idx}".encode()).hexdigest())
    return ids


def ensure_index(pc: Pinecone, settings: Settings, dim: int, recreate: bool) -> None:
    """Create the index with the model's real dimension; refuse to write into a mismatched one."""
    name = settings.pinecone_index_name
    if recreate and pc.has_index(name):
        logger.info("Deleting index %s", name)
        pc.delete_index(name)
    if pc.has_index(name):
        check_index_dimension(pc.describe_index(name).dimension, dim, settings.embedding_model, name)
    else:
        logger.info("Creating index %s (dim=%d, metric=cosine)", name, dim)
        pc.create_index(
            name=name,
            dimension=dim,
            metric="cosine",
            spec=ServerlessSpec(cloud=settings.pinecone_cloud, region=settings.pinecone_region),
        )
    while not pc.describe_index(name).status["ready"]:
        time.sleep(1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--chunk-size", type=int, default=500)
    parser.add_argument("--chunk-overlap", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--recreate", action="store_true", help="Delete and rebuild the index first.")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    settings = get_settings()

    docs = normalise_metadata(load_pdfs(args.data_dir))
    if not docs:
        raise SystemExit(f"No PDFs found in {args.data_dir}")
    chunks = split(docs, args.chunk_size, args.chunk_overlap)
    logger.info("Loaded %d pages -> %d chunks", len(docs), len(chunks))

    embeddings = build_embeddings(settings)  # same factory the API uses at query time
    dim = embedding_dimension(embeddings)
    logger.info("Embedding model %s -> %d-d vectors", settings.embedding_model, dim)

    pc = Pinecone(api_key=settings.pinecone_api_key.get_secret_value())
    ensure_index(pc, settings, dim, args.recreate)

    from langchain_pinecone import PineconeVectorStore

    store = PineconeVectorStore(index=pc.Index(settings.pinecone_index_name), embedding=embeddings)
    ids = chunk_ids(chunks)
    for start in range(0, len(chunks), args.batch_size):
        end = start + args.batch_size
        store.add_documents(chunks[start:end], ids=ids[start:end])
        logger.info("Upserted %d/%d", min(end, len(chunks)), len(chunks))

    logger.info("Done. Index %s is ready.", settings.pinecone_index_name)


if __name__ == "__main__":
    main()
