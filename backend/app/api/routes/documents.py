from fastapi import APIRouter, UploadFile, File, HTTPException
from app.models.schemas import UploadResponse, ErrorResponse
from app.services import extraction, rag
import uuid
from pathlib import Path
from datetime import datetime, timezone
from app.core.config import settings

router = APIRouter()

ACCEPTED_MIME_TYPES = {
    "application/pdf",
    "image/png",
    "image/jpeg",
    "image/tiff",
    "image/webp",
}


@router.post(
    "/upload",
    response_model=UploadResponse,
    responses={400: {"model": ErrorResponse}, 422: {"model": ErrorResponse}},
    summary="Upload a document",
    description="Accepts a PDF or image file, extracts text, chunks and indexes it. Returns a document_id for subsequent Q&A requests.",
)
async def upload_document(file: UploadFile = File(...)):
    if file.content_type not in ACCEPTED_MIME_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{file.content_type}'. Accepted: PDF, PNG, JPEG, TIFF, WEBP.",
        )

    document_id = str(uuid.uuid4())
    # Persist raw file so extraction service can read it
    ext = Path(file.filename).suffix or ".bin"
    save_path = settings.UPLOAD_DIR / f"{document_id}{ext}"
    content = await file.read()
    save_path.write_bytes(content)

    # Extract text
    try:
        extracted_text = extraction.extract(save_path, file.content_type)
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Text extraction failed: {e}")

    if not any(extracted_text.strip()):
        raise HTTPException(
            status_code=422, detail="No readable text found in document."
        )

    # Chunk + Embed + Index text
    num_chunks = await rag.index_document(
        extracted_text=extracted_text, document_id=document_id
    )

    return UploadResponse(
        document_id=document_id,
        filename=file.filename,
        num_chunks=num_chunks,
        created_at=datetime.now(timezone.utc),
    )
