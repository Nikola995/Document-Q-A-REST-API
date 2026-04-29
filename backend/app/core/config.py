from pydantic_settings import BaseSettings
from pathlib import Path
import torch


class Settings(BaseSettings):
    PROJECT_NAME: str = "Document Q&A API"
    VERSION: str = "1.0.0"

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

    class Config:
        env_file = ".env"
        case_sensitive = True


settings = Settings()
