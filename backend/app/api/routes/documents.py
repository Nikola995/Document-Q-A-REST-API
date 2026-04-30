from fastapi import APIRouter, UploadFile, File, HTTPException, Request
from app.models.schemas import UploadResponse, ErrorResponse
from app.services import extraction, rag
import uuid
from pathlib import Path
from datetime import datetime, timezone
from app.services.cache import get_cached_document_id, set_cached_document_id
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
async def upload_document(request: Request, file: UploadFile = File(...)):
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

    text = extraction.extract(save_path, file.content_type)
    # Check cache initially if the exact document (its extracted text) was already processed
    cached_document_id = await get_cached_document_id(text, request)
    if cached_document_id is not None:
        save_path.unlink(
            missing_ok=True
        )  # pathlib's delete, missing_ok avoids FileNotFoundError
        return UploadResponse(
            document_id=cached_document_id, filename=file.filename, already_exists=True
        )
    # Chunk + Embed + Index text
    await rag.index_document(extracted_text=extracted_text, document_id=document_id)
    # Add the newly processed document id to cache
    await set_cached_document_id(extracted_text, document_id, request)

    return UploadResponse(
        document_id=document_id,
        filename=file.filename,
        created_at=datetime.now(timezone.utc),
    )
