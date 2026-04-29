from pydantic_settings import BaseSettings
from pathlib import Path
import torch


class Settings(BaseSettings):
    PROJECT_NAME: str = "Document Q&A API"
    VERSION: str = "0.1.0"

    # CORS
    ALLOWED_ORIGINS: list[str] = ["http://localhost:8501"]  # Streamlit default port

    # LLM
    qa_model_name: str = "distilbert-base-uncased-distilled-squad"

    # Storage — where uploaded files and FAISS indexes are persisted
    INDEX_DIR: Path = Path("/tmp/indexes")
    UPLOAD_DIR: Path = Path("/tmp/uploads")

    class Config:
        env_file = ".env"
        case_sensitive = True


settings = Settings()
