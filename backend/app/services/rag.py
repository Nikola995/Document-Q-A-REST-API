from fastapi import Request
from chonkie.pipeline import Pipeline
from chonkie import Document
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import pickle
from pathlib import Path
from app.services.cache import get_cached_embedding, set_cached_embedding
from app.models.internal import ChunkMetadata, DocumentMetadata, FileProcessResult
from app.models.schemas import ContextChunk
from app.core.config import settings


def load_embedding_model() -> dict:
    return {"model": SentenceTransformer(settings.EMBEDDING_MODEL)}


def session_exists(session_id: str) -> bool:
    base = settings.INDEX_DIR / session_id
    return (
        base.with_suffix(".faiss").exists()
        and base.with_suffix(".meta").exists()
        and base.with_suffix(".docs").exists()
    )


def _load_or_create_index(base_path: Path, dim: int) -> faiss.IndexFlatIP:
    faiss_path = base_path.with_suffix(".faiss")
    if faiss_path.exists():
        return faiss.read_index(str(faiss_path))
    return faiss.IndexFlatIP(dim)


def _load_or_create_chunks_store(base_path: Path) -> list[ChunkMetadata]:
    meta_path = base_path.with_suffix(".meta")
    if meta_path.exists():
        with open(meta_path, "rb") as f:
            return pickle.load(f)
    return []


def _load_or_create_docs_store(base_path: Path) -> dict[str, DocumentMetadata]:
    docs_path = base_path.with_suffix(".docs")
    if docs_path.exists():
        with open(docs_path, "rb") as f:
            return pickle.load(f)
    return {}


def _chunk(text: str) -> Document:
    """A RAG ingestion pipeline, chunking the text of a single document,
    intended to be followed by creating embeddings for each chunk and
    indexing into a vector database.

    Args:
        text (str): The entire text of a single document

    Returns:
        Document: Document type from chonkie library, which contains the chunks.
    """
    doc = (
        Pipeline()
        .process_with("markdown")
        .chunk_with("semantic", threshold=0.8, chunk_size=settings.CHUNK_SIZE)
        .refine_with("overlap", context_size=settings.CHUNK_OVERLAP)
        .run(texts=text)
    )
    return doc


def _embed(doc: Document, model) -> np.ndarray:
    # FAISS requires float32 specifically and will throw a cryptic error if it receives float64, which is numpy's default.
    return np.vstack([model.encode(chunk.text) for chunk in doc.chunks]).astype(
        np.float32
    )


async def index_documents(
    session_id: str, file_results: list[FileProcessResult], request: Request
) -> None:

    base_path = settings.INDEX_DIR / session_id
    # load or create stores
    chunks_store: list[ChunkMetadata] = _load_or_create_chunks_store(base_path)
    docs_store: dict[str, DocumentMetadata] = _load_or_create_docs_store(base_path)

    model = request.app.state.embedding_model["model"]
    new_embeddings = []

    # process all non-duplicate files
    for result in file_results:
        if result.already_exists:
            continue

        doc = _chunk(result.text)
        chunk_embeddings = _embed(doc, model)
        # extend chunk list — order preserved, matches FAISS index position
        chunks_store.extend(
            [
                ChunkMetadata(document_id=result.document_id, text=chunk.text)
                for chunk in doc.chunks
            ]
        )
        # update document store
        docs_store[result.document_id] = DocumentMetadata(
            document_id=result.document_id, filename=result.filename
        )

        new_embeddings.append(chunk_embeddings)

    if not new_embeddings:
        return  # all files were duplicates, nothing to index

    # build and save index once after all files processed
    combined_embeddings = np.vstack(new_embeddings)
    faiss.normalize_L2(combined_embeddings)

    index = _load_or_create_index(base_path, combined_embeddings.shape[1])
    index.add(combined_embeddings)

    # persist all three stores once
    faiss.write_index(index, str(base_path.with_suffix(".faiss")))
    with open(base_path.with_suffix(".meta"), "wb") as f:
        pickle.dump(chunks_store, f)
    with open(base_path.with_suffix(".docs"), "wb") as f:
        pickle.dump(docs_store, f)
    return None


async def retrieve_chunks(
    session_id: str, question: str, request: Request
) -> ContextChunk:
    if not session_exists(session_id=session_id):
        raise FileNotFoundError

    base_path = settings.INDEX_DIR / session_id
    # Load stores
    index = faiss.read_index(str(base_path.with_suffix(".faiss")))
    with open(base_path.with_suffix(".meta"), "rb") as f:
        chunks_store: list[ChunkMetadata] = pickle.load(f)
    with open(base_path.with_suffix(".docs"), "rb") as f:
        docs_store: dict[str, DocumentMetadata] = pickle.load(f)

    model = request.app.state.embedding_model["model"]
    # check embedding cache
    cached = await get_cached_embedding(question, request)
    if cached is None:
        print("Cache MISS!")
        q_emb = model.encode([question], convert_to_numpy=True).astype(np.float32)
        faiss.normalize_L2(q_emb)
        await set_cached_embedding(question, q_emb, request)
    else:
        print("Cache HIT!")
        q_emb = cached

    faiss.normalize_L2(q_emb)
    scores, indices = index.search(q_emb, settings.TOP_K_CHUNKS)

    top_idx = indices[0][0]
    top_score = float(scores[0][0])
    chunk = chunks_store[top_idx]
    doc = docs_store[chunk.document_id]

    return ContextChunk(text=chunk.text, score=top_score, filename=doc.filename)
