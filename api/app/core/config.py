from pathlib import Path
from functools import lru_cache

from langchain.embeddings import init_embeddings
from pydantic_settings import BaseSettings, SettingsConfigDict
from qdrant_client import QdrantClient

ENV_FILE = Path(_file_).resolve().parents[2] / ".env"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="RAG_", env_file=ENV_FILE, extra="ignore")

    app_name: str = "clause"
    data_path: str = "data/shop_handbook.pdf"

    chat_provider: str = "ollama"
    chat_model: str = "gemma3:1b"
    temperature: float = 0.1

    embedding_provider: str = "ollama"
    embedding_model: str = "nomic-embed-text:latest"
    embed_dim: int = 768

    chunk_size: int = 600
    chunk_overlap: int = 100

    collection_name: str = "shop_handbook"
    qdrant_data: str = "./qdrant_data"

    system: str = "you are a helpful AI clause shop assistant!"


@lru_cache
def get_settings() -> Settings:
    return Settings()


def get_embeddings(settings: Settings | None = None):
    settings = settings or get_settings()
    return init_embeddings(settings.embedding_model, provider=settings.embedding_provider)


def get_qdrant_client(settings: Settings | None = None) -> QdrantClient:
    settings = settings or get_settings()
    return QdrantClient(path=settings.qdrant_data)