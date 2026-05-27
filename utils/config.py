import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


load_dotenv()


@dataclass(frozen=True)
class AppConfig:
    """Central app settings loaded from environment variables."""

    project_root: Path = Path(__file__).resolve().parents[1]
    upload_dir: Path = project_root / "data" / "uploads"
    sample_dir: Path = project_root / "data" / "samples"
    history_db_path: Path = project_root / "data" / "history.db"
    vectorstore_dir: Path = project_root / "vectorstore"
    collection_name: str = os.getenv("CHROMA_COLLECTION_NAME", "document_assistant")
    llm_provider: str = os.getenv("LLM_PROVIDER", "openai").lower()
    openai_model: str = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    openai_embedding_model: str = os.getenv(
        "OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"
    )
    ollama_base_url: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    ollama_model: str = os.getenv("OLLAMA_MODEL", "llama3.1")
    ollama_embedding_model: str = os.getenv(
        "OLLAMA_EMBEDDING_MODEL", "nomic-embed-text"
    )
    chunk_size: int = int(os.getenv("CHUNK_SIZE", "1000"))
    chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", "180"))
    top_k: int = int(os.getenv("TOP_K", "5"))
    use_deepeval: bool = os.getenv("USE_DEEPEVAL", "false").lower() == "true"


def get_config() -> AppConfig:
    config = AppConfig()
    config.upload_dir.mkdir(parents=True, exist_ok=True)
    config.sample_dir.mkdir(parents=True, exist_ok=True)
    config.vectorstore_dir.mkdir(parents=True, exist_ok=True)
    return config
