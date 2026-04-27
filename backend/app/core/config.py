from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    PROJECT_NAME: str = "Document Q&A API"
    VERSION: str = "0.1.0"

    # CORS
    ALLOWED_ORIGINS: list[str] = ["http://localhost:8501"]  # Streamlit default port
    
    # Storage — where uploaded files and FAISS indexes are persisted
    UPLOAD_DIR: str = "/tmp/uploads"

    class Config:
        env_file = ".env"
        case_sensitive = True


settings = Settings()
