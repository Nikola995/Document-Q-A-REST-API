from chonkie.pipeline import Pipeline
from chonkie import Document
from pathlib import Path
from app.core.config import settings


def _chunk_text(extracted_text: str, doc_path: Path) -> Document:
    """A RAG ingestion pipeline, chunking the text of a single document,
    intended for indexing into a vector database.

    Args:
        extracted_text (str): The entire text of a single document

    Returns:
        Document: Document type from chonkie library, relevant are the chunks of the document text.
    """
    doc = (
        Pipeline()
        .process_with("markdown")
        .chunk_with("semantic", threshold=0.8, chunk_size=512)
        .refine_with("overlap", context_size=100)
        .export_with(porter_type="json", file=doc_path)
        .run(texts=extracted_text)
    )
    return doc


async def index_document(extracted_text: str, doc_path: Path) -> int:
    doc = _chunk_text(extracted_text=extracted_text, doc_path=doc_path)
    return len(doc.chunks)


async def retrieve_chunks(document_id: str, question: str) -> ...:
    raise NotImplementedError
