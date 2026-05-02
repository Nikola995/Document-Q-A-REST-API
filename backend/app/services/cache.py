from fastapi import Request
from typing import Optional
import numpy as np
import hashlib
from loguru import logger
from app.core.config import settings


# --- Cache layer ---
async def _get_cache(key: str, request: Request) -> Optional[bytes]:
    try:
        return await request.app.state.redis.get(key)
    except Exception:
        logger.warning("Redis unavailable, automatic cache miss")
        return None


async def _set_cache(key: str, value: bytes, request: Request) -> None:
    try:
        await request.app.state.redis.setex(key, settings.CACHE_TTL_SECONDS, value)
    except Exception:
        logger.warning(
            f"Redis unavailable, could not cache key",
            extra={"key": key},
        )


# --- Key generation ---
def _document_key(session_id: str, text: str) -> str:
    return hashlib.sha256(f"{session_id}:{text}".encode()).hexdigest()


def _embedding_key(question: str) -> str:
    return hashlib.sha256(question.encode()).hexdigest()


# --- Document caching ---
async def get_cached_document_id(
    session_id: str, text: str, request: Request
) -> str | None:
    raw = await _get_cache(_document_key(session_id, text), request)
    logger.info(
        f"cache_{"hit" if raw else "miss"}",
        extra={
            "type": "document",
            "document_id": raw.decode() if raw else "",
            "session_id": session_id,
        },
    )
    return raw.decode() if raw else None


async def set_cached_document_id(
    session_id: str, text: str, document_id: str, request: Request
) -> None:
    logger.info(
        "cache_set",
        extra={
            "type": "document",
            "document_id": document_id,
            "session_id": session_id,
        },
    )
    await _set_cache(_document_key(session_id, text), document_id.encode(), request)


# --- QA Question Embedding caching ---
async def get_cached_embedding(question: str, request: Request) -> Optional[np.ndarray]:
    raw = await _get_cache(_embedding_key(question), request)
    logger.info(
        f"cache_{"hit" if raw else "miss"}",
        extra={
            "type": "question",
            "question": question,
        },
    )
    return np.frombuffer(raw, dtype=np.float32).reshape(1, -1) if raw else None


async def set_cached_embedding(
    question: str, embedding: np.ndarray, request: Request
) -> None:
    logger.info(
        "cache_set",
        extra={
            "type": "question",
            "question": question,
        },
    )
    await _set_cache(_embedding_key(question), embedding.tobytes(), request)
