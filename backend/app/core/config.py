from pydantic_settings import BaseSettings
from pathlib import Path


class Settings(BaseSettings):
    PROJECT_NAME: str = "Document Q&A API"
    VERSION: str = "3.0.0"

    # CORS
    ALLOWED_ORIGINS: list[str] = ["http://localhost:8501"]  # Streamlit default port

    # LLM
    QA_MODEL: str # no default — must be set in .env

    # Storage — where uploaded files and FAISS indexes are persisted
    INDEX_DIR: Path = Path("/tmp/indexes")
    UPLOAD_DIR: Path = Path("/tmp/uploads")

    # Chunking + Embedding
    CHUNK_SIZE: int = 512
    CHUNK_OVERLAP: int = 100
    EMBEDDING_MODEL: str # no default — must be set in .env
    TOP_K_CHUNKS: int = 3

    # Caching
    REDIS_URL: str = "redis://localhost:6379"  # local dev fallback only
    CACHE_TTL_SECONDS: int = 3600  # 1h expiration time

    # LLM
    NER_MODEL: str # no default — must be set in .env
    NER_LABELS: list[str] = [
        "person",
        "organization",
        "date",
        "due date",
        "location",
        "invoice number",
        "line item",
        "amount",
        "tax",
        "payment terms",
        "obligation",
        "jurisdiction",
    ]

    class Config:
        env_file = ".env"
        case_sensitive = True


settings = Settings()
