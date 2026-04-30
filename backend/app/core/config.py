from pydantic_settings import BaseSettings
from pathlib import Path
import torch


class Settings(BaseSettings):
    PROJECT_NAME: str = "Document Q&A API"
    VERSION: str = "2.1.0"

    # CORS
    ALLOWED_ORIGINS: list[str] = ["http://localhost:8501"]  # Streamlit default port

    # LLM
    QA_MODEL: str = "distilbert-base-uncased-distilled-squad"

    # Storage — where uploaded files and FAISS indexes are persisted
    INDEX_DIR: Path = Path("/tmp/indexes")
    UPLOAD_DIR: Path = Path("/tmp/uploads")
    
    # Chunking + Embedding
    CHUNK_SIZE: int = 512
    CHUNK_OVERLAP: int = 100
    EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"
    TOP_K_CHUNKS: int = 3
    
    # Caching
    REDIS_URL: str = "redis://localhost:6379"  # safe default for local dev only
    CACHE_TTL_SECONDS: int = 3600 # 1h expiration time

    class Config:
        env_file = ".env"
        case_sensitive = True


settings = Settings()
