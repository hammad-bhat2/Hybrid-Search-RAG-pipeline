import os
from typing import Optional
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

    APP_NAME: str = "Hybrid Search RAG Pipeline"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = False

    # Qdrant Database Configuration
    QDRANT_HOST: str = os.getenv("QDRANT_HOST", "localhost")
    QDRANT_PORT: int = int(os.getenv("QDRANT_PORT", 6333))
    QDRANT_URL: Optional[str] = os.getenv("QDRANT_URL", None)
    QDRANT_API_KEY: Optional[str] = os.getenv("QDRANT_API_KEY", None)
    QDRANT_STORAGE_PATH: str = os.getenv("QDRANT_STORAGE_PATH", "./data/qdrant_db")
    COLLECTION_NAME: str = os.getenv("COLLECTION_NAME", "rag_hybrid_collection")

    # Embeddings Configuration
    DENSE_MODEL_NAME: str = os.getenv("DENSE_MODEL_NAME", "BAAI/bge-small-en-v1.5")
    DENSE_VECTOR_SIZE: int = 384
    SPARSE_MODEL_NAME: str = os.getenv("SPARSE_MODEL_NAME", "Qdrant/bm25")

    # Retrieval Configuration
    RRF_K: int = int(os.getenv("RRF_K", 60))
    DENSE_WEIGHT: float = float(os.getenv("DENSE_WEIGHT", 1.0))
    SPARSE_WEIGHT: float = float(os.getenv("SPARSE_WEIGHT", 1.0))
    DEFAULT_TOP_K: int = int(os.getenv("DEFAULT_TOP_K", 5))

    # Ingestion Configuration
    CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", 500))
    CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", 50))
    UPLOAD_DIR: str = os.getenv("UPLOAD_DIR", "data/raw")

    # LLM Generation Configuration
    LLM_PROVIDER: str = os.getenv("LLM_PROVIDER", "auto")  # 'auto', 'openai', 'gemini', 'fallback'
    OPENAI_API_KEY: Optional[str] = os.getenv("OPENAI_API_KEY", "")
    OPENAI_MODEL: str = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    GEMINI_API_KEY: Optional[str] = os.getenv("GEMINI_API_KEY", "")
    GEMINI_MODEL: str = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
    LLM_TEMPERATURE: float = 0.1
    LLM_MAX_TOKENS: int = 1024

settings = Settings()