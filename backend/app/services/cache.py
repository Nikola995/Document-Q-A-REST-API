from fastapi import Request
from typing import Optional
import numpy as np
import hashlib
from app.core.config import settings


# --- Cache layer ---
async def _get_cache(key: str, request: Request) -> Optional[bytes]:
    try:
        return await request.app.state.redis.get(key)
    except Exception:
        # TODO: switch to logger.warning when implementing structured logging
        print("Redis unavailable, cache miss")
        return None


async def _set_cache(key: str, value: bytes, request: Request) -> None:
    try:
        await request.app.state.redis.setex(key, settings.CACHE_TTL_SECONDS, value)
    except Exception:
        # TODO: switch to logger.warning when implementing structured logging
        print(f"Redis unavailable, could not cache key: {key}")


# --- Key generation ---
def _document_key(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def _embedding_key(question: str) -> str:
    return hashlib.sha256(question.encode()).hexdigest()


# --- Document caching ---
async def get_cached_document_id(text: str, request: Request) -> Optional[str]:
    raw = await _get_cache(_document_key(text), request)
    return raw.decode() if raw else None


async def set_cached_document_id(text: str, document_id: str, request: Request) -> None:
    await _set_cache(_document_key(text), document_id.encode(), request)


# --- QA Question Embedding caching ---
async def get_cached_embedding(question: str, request: Request) -> Optional[np.ndarray]:
    raw = await _get_cache(_embedding_key(question), request)
    return np.frombuffer(raw, dtype=np.float32).reshape(1, -1) if raw else None


async def set_cached_embedding(
    question: str, embedding: np.ndarray, request: Request
) -> None:
    await _set_cache(_embedding_key(question), embedding.tobytes(), request)
