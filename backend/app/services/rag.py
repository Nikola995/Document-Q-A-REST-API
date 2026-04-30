from fastapi import Request
from chonkie.pipeline import Pipeline
from chonkie import Document
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import pickle
from pathlib import Path
from app.services.cache import get_cached_embedding, set_cached_embedding
from app.models.schemas import SourceChunk
from app.core.config import settings


def _chunking_and_embedding(extracted_text: str) -> Document:
    """A RAG ingestion pipeline, chunking the text of a single document,
    followed by creating embeddings for each chunk,
    intended for indexing into a vector database.

    Args:
        extracted_text (str): The entire text of a single document

    Returns:
        Document: Document type from chonkie library, which contains the chunks.
    """
    # TODO: Initialize embedding model in lifespan and then inject in pipeline
    doc = (
        Pipeline()
        .process_with("markdown")
        .chunk_with("semantic", threshold=0.8, chunk_size=settings.CHUNK_SIZE)
        .refine_with("overlap", context_size=settings.CHUNK_OVERLAP)
        .refine_with("embeddings", embedding_model=settings.EMBEDDING_MODEL)
        .run(texts=extracted_text)
    )
    return doc


async def index_document(extracted_text: str, document_id: str) -> None:
    doc = _chunking_and_embedding(extracted_text=extracted_text)
    embeddings = [chunk.embedding for chunk in doc.chunks]
    # FAISS requires float32 specifically and will throw a cryptic error if it receives float64, which is numpy's default.
    embeddings = np.vstack([chunk.embedding for chunk in doc.chunks]).astype(np.float32)
    # Create index
    index = faiss.IndexFlatIP(
        embeddings.shape[1]
    )  # inner-product ≈ cosine on normalized vecs
    faiss.normalize_L2(embeddings)
    index.add(embeddings)

    doc_path = settings.INDEX_DIR / document_id
    faiss.write_index(index, str(doc_path.with_suffix(".faiss")))
    chunk_texts = [chunk.text for chunk in doc.chunks]
    with open(doc_path.with_suffix(".meta"), "wb") as f:
        pickle.dump(chunk_texts, f)
    return None


async def retrieve_chunks(
    document_id: str, question: str, request: Request
) -> SourceChunk:
    doc_path = settings.INDEX_DIR / document_id
    # If either the .faiss or .meta file is missing, raise FileNotFoundError
    if not (
        doc_path.with_suffix(".faiss").exists()
        and doc_path.with_suffix(".meta").exists()
    ):
        raise FileNotFoundError
    # Load both files
    index = faiss.read_index(str(doc_path.with_suffix(".faiss")))
    with open(doc_path.with_suffix(".meta"), "rb") as f:
        chunk_texts = pickle.load(f)

    # Retrieve top-k chunks
    model = SentenceTransformer(settings.EMBEDDING_MODEL)
    # Check if cache exists to optimize set_cached_embedding operation
    cached = await get_cached_embedding(question=question, request=request)
    if cached is None:
        print("Cache MISS!")
        q_emb = model.encode([question], convert_to_numpy=True)
        await set_cached_embedding(question=question, embedding=q_emb, request=request)
    else:
        print("Cache HIT!")
        q_emb = cached
    faiss.normalize_L2(q_emb)
    scores, indices = index.search(q_emb, settings.TOP_K_CHUNKS)
    top_k_chunks = [
        SourceChunk(text=chunk_texts[i], score=float(scores[0][j]))
        for j, i in enumerate(indices[0])
        if i != -1
    ]
    print(top_k_chunks)
    return top_k_chunks[0]
