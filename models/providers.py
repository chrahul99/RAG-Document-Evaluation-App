import os
from urllib.error import URLError
from urllib.request import urlopen

from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from utils.config import AppConfig


def provider_setup_message(config: AppConfig) -> str | None:
    """Return a setup message when the selected model provider is not ready."""
    if config.llm_provider == "openai" and not os.getenv("OPENAI_API_KEY"):
        return (
            "OpenAI is selected, but OPENAI_API_KEY is missing. Add your key to "
            "the .env file, then restart Streamlit. Or set LLM_PROVIDER=ollama "
            "if you want to use local Ollama models."
        )
    if config.llm_provider == "ollama":
        try:
            with urlopen(f"{config.ollama_base_url}/api/tags", timeout=1):
                return None
        except (OSError, URLError):
            return (
                "Ollama is selected, but the Ollama server is not running. "
                "Install Ollama, open the Ollama app or run `ollama serve`, "
                "then pull `llama3.1` and `nomic-embed-text`."
            )
    return None


def get_embeddings(config: AppConfig):
    """Return the embedding model configured for OpenAI or Ollama."""
    if config.llm_provider == "ollama":
        return OllamaEmbeddings(
            model=config.ollama_embedding_model,
            base_url=config.ollama_base_url,
        )

    return OpenAIEmbeddings(model=config.openai_embedding_model)


def get_chat_model(config: AppConfig):
    """Return the chat model configured for OpenAI or Ollama."""
    if config.llm_provider == "ollama":
        return ChatOllama(
            model=config.ollama_model,
            base_url=config.ollama_base_url,
            temperature=0.1,
        )

    return ChatOpenAI(model=config.openai_model, temperature=0.1)
