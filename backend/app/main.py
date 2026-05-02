from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import redis.asyncio as aioredis
import torch
from loguru import logger
import sys
from pathlib import Path
from app.api.routes import documents, qa
from app.services.rag import load_embedding_model
from app.services.llm import load_qa_model
from app.services.ner import load_ner_model
from app.core.config import settings


def setup_logging() -> None:
    # remove default stderr handler
    logger.remove()
    Path("logs").mkdir(exist_ok=True)
    # add structured JSON handler to file — for monitoring/profiling
    logger.add(
        "logs/app.log",
        serialize=True,  # outputs as JSON
        rotation="10 MB",
        retention="7 days",
        level="INFO",
        enqueue=True,  # async-safe, important for FastAPI
    )
    # add human-readable handler to stderr — for development
    logger.add(
        sys.stderr,
        level="DEBUG",
        colorize=True,
        format="<green>{time:HH:mm:ss}</green> | <level>{level}</level> | {message}",
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    setup_logging()
    logger.info("Starting application")
    # Ensure directories exist
    logger.info("Creating storage directories")
    settings.UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    settings.INDEX_DIR.mkdir(parents=True, exist_ok=True)
    # Load the Redis client
    logger.info("Connecting to Redis")
    app.state.redis = aioredis.from_url(settings.REDIS_URL)
    try:
        await app.state.redis.ping()
        logger.info("Redis connection established")
    except Exception as e:
        logger.warning(f"Redis unavailable at startup: {e}")
    # Load the models once at app start
    app.state.embedding_model = load_embedding_model()
    app.state.qa_model = load_qa_model()
    app.state.ner_model = load_ner_model()
    logger.info("All models loaded — application ready")
    yield
    # shutdown
    logger.info("Shutting down application")
    del app.state.embedding_model
    del app.state.qa_model
    del app.state.ner_model
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
