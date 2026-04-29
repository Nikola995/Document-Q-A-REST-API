from chonkie.pipeline import Pipeline
from chonkie import Document
import numpy as np
import faiss
import pickle
from pathlib import Path
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


async def index_document(extracted_text: str, document_id: str) -> int:
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
    return len(doc.chunks)


async def retrieve_chunks(document_id: str, question: str) -> ...:
    raise NotImplementedError
