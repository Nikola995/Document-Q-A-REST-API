from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import redis.asyncio as aioredis
import torch

from app.api.routes import documents, qa
from app.services.llm import load_model
from app.core.config import settings


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Ensure directories exist
    settings.UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    settings.INDEX_DIR.mkdir(parents=True, exist_ok=True)
    # Load the Redis client
    app.state.redis = aioredis.from_url(settings.REDIS_URL)
    try:
        await app.state.redis.ping()
        # TODO: change to logs when implementing structured logging
        print("Redis connection established")
    except Exception as e:
        print(f"Redis unavailable at startup: {e}")
    # Load the LLM model once at app start
    app.state.qa_model = load_model()
    yield
    del app.state.qa_model
    if torch.cuda.is_available():
        torch.cuda.synchronize()  # wait for all CUDA ops to finish
        torch.cuda.empty_cache()


app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    description="Document Q&A API — upload documents and ask questions about their content.",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(documents.router, prefix="/api/v1", tags=["documents"])
app.include_router(qa.router, prefix="/api/v1", tags=["qa"])


@app.get("/health", tags=["health"])
async def health_check():
    return {"status": "ok", "version": settings.VERSION}
