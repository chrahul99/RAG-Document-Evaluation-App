import time
from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.documents import Document

from models.providers import get_embeddings
from utils.config import AppConfig
from utils.logger import get_logger


logger = get_logger(__name__)


def get_vector_store(config: AppConfig) -> Chroma:
    """Connect to the persistent Chroma vector database."""
    embeddings = get_embeddings(config)
    return Chroma(
        collection_name=config.collection_name,
        embedding_function=embeddings,
        persist_directory=str(config.vectorstore_dir),
    )


def add_documents_to_vector_store(
    chunks: list[Document], config: AppConfig
) -> dict[str, float | int]:
    """Embed and persist chunks in ChromaDB."""
    if not chunks:
        return {"chunks_added": 0, "embedding_time_seconds": 0.0}

    start_time = time.perf_counter()
    vector_store = get_vector_store(config)
    vector_store.add_documents(chunks)
    elapsed = time.perf_counter() - start_time
    logger.info("Added %s chunks to vector store", len(chunks))
    return {"chunks_added": len(chunks), "embedding_time_seconds": elapsed}


def retrieve_relevant_chunks(
    question: str, config: AppConfig, top_k: int | None = None
) -> dict:
    """Retrieve chunks with relevance scores from ChromaDB."""
    start_time = time.perf_counter()
    vector_store = get_vector_store(config)
    results = vector_store.similarity_search_with_relevance_scores(
        question,
        k=top_k or config.top_k,
    )
    elapsed = time.perf_counter() - start_time

    sources = []
    for document, score in results:
        sources.append(
            {
                "content": document.page_content,
                "score": float(score),
                "source": document.metadata.get("source", "Unknown"),
                "page": document.metadata.get("page", "Unknown"),
                "chunk_index": document.metadata.get("chunk_index", "Unknown"),
            }
        )

    return {
        "sources": sources,
        "retrieval_latency_seconds": elapsed,
        "average_relevance_score": (
            sum(item["score"] for item in sources) / len(sources) if sources else 0.0
        ),
    }


def vector_store_exists(config: AppConfig) -> bool:
    """Check whether Chroma has files on disk."""
    path = Path(config.vectorstore_dir)
    if not path.exists():
        return False
    return any(item.name != ".gitkeep" for item in path.iterdir())
