from fastapi import APIRouter, UploadFile, File, HTTPException, Request, Form
from app.models.internal import FileProcessResult
from app.models.schemas import FileUploadResult, UploadResponse, ErrorResponse
from app.services import extraction, rag
import uuid
import asyncio
from loguru import logger
import time
from typing import Optional
from pathlib import Path
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


async def process_file(
    file: UploadFile, session_id: str, request: Request
) -> FileProcessResult:
    # Persist raw file so extraction service can read it
    document_id = str(uuid.uuid4())
    content = await file.read()
    save_path = (
        settings.UPLOAD_DIR / f"{document_id}{Path(file.filename).suffix or '.bin'}"
    )
    save_path.write_bytes(content)
    # Call extraction service
    try:
        extraction_time_start = time.perf_counter()
        extracted_text = extraction.extract(save_path, file.content_type)
        extraction_time_total = time.perf_counter() - extraction_time_start
        logger.info(
            "extraction_success",
            extra={
                "filename": file.filename,
                "session_id": session_id,
                "extracted_text_len": len(extracted_text),
                "execution_time_s": extraction_time_total,
            },
        )
    except Exception as e:
        logger.error(
            "extraction_failure",
            extra={
                "filename": file.filename,
                "session_id": session_id,
                "error": str(e),
            },
        )
        raise HTTPException(
            status_code=422,
            detail=f"Text extraction failed for document {file.filename}: {e}",
        )

    if not any(extracted_text.strip()):
        logger.error(
            "extraction_failure",
            extra={
                "filename": file.filename,
                "session_id": session_id,
                "error": "No readable text found",
            },
        )
        raise HTTPException(
            status_code=422,
            detail=f"No readable text found in document {file.filename}.",
        )
    # Check cache if document with same content was already uploaded (for current session)
    cached = await get_cached_document_id(session_id, extracted_text, request)
    already_exists = cached is not None
    if already_exists:
        save_path.unlink(missing_ok=True)
        file_process_result = FileProcessResult(
            document_id=cached,
            filename=file.filename,
            text="",
            already_exists=already_exists,
        )
    else:
        # If cache miss, add document content to cache (for current session)
        await set_cached_document_id(session_id, extracted_text, document_id, request)
        file_process_result = FileProcessResult(
            document_id=document_id,
            filename=file.filename,
            text=extracted_text,
            already_exists=already_exists,
        )
    logger.info(
        "file_processed",
        extra={
            "filename": file.filename,
            "session_id": session_id,
            "already_exists": already_exists,
        },
    )
    return file_process_result


@router.post(
    "/upload",
    response_model=UploadResponse,
    responses={400: {"model": ErrorResponse}, 422: {"model": ErrorResponse}},
    summary="Upload a document",
    description="Accepts a PDF or image file, extracts text, chunks and indexes it. Returns a document_id for subsequent Q&A requests.",
)
async def upload_document(
    request: Request,
    session_id: Optional[str] = Form(default=None),
    files: list[UploadFile] = File(...),
):
    # on upload start
    upload_time_start = time.perf_counter()
    logger.info(
        "upload_received",
        extra={
            "file_count": len(files),
            "filenames": [f.filename for f in files],
            "session_id": session_id,
            "is_new_session": session_id is None,
        },
    )
    # validate all files before processing any
    for file in files:
        if file.content_type not in ACCEPTED_MIME_TYPES:
            logger.error(
                "unsupported_file_type",
                extra={
                    "filename": file.filename,
                    "type": file.content_type,
                    "session_id": session_id,
                },
            )
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file type '{file.content_type}' for file '{file.filename}'. "
                f"Accepted: PDF, PNG, JPEG, TIFF, WEBP.",
            )
    # validate session exists if provided
    if session_id is None:
        session_id = str(uuid.uuid4())
    elif not rag.session_exists(session_id):
        logger.error(
            "invalid_session_id",
            extra={
                "session_id": session_id,
                "endpoint": "/upload",
            },
        )
        raise HTTPException(
            status_code=404, detail=f"Session '{session_id}' not found."
        )
    # concurrent extraction and cache checks
    file_results = await asyncio.gather(
        *[process_file(f, session_id, request) for f in files], return_exceptions=True
    )
    # check for any exceptions after all tasks complete
    errors = [r for r in file_results if isinstance(r, Exception)]
    if errors:
        # clean up any successful results that were already processed (to not leave orphaned files)
        for result in file_results:
            if isinstance(result, FileProcessResult):
                (
                    settings.UPLOAD_DIR
                    / f"{result.document_id}{Path(result.filename).suffix}"
                ).unlink(missing_ok=True)
        # and finally return the full list of errors as a HTTPException
        raise HTTPException(status_code=422, detail=[str(e) for e in errors])

    # sequential (Chunk + Embed + Index) to avoid collisions
    try:
        await rag.index_documents(session_id, list(file_results), request)
    except Exception as e:
        logger.error(
            "indexing_failure",
            extra={
                "session_id": session_id,
                "error": str(e),
            },
        )
        raise HTTPException(
            status_code=422,
            detail=f"Indexing failed for session {session_id}: {e}",
        )
    # on upload end
    upload_time_total = time.perf_counter() - upload_time_start
    logger.info(
        "upload_completed",
        extra={
            "session_id": session_id,
            "files_indexed": len([r for r in file_results if not r.already_exists]),
            "duplicates": len([r for r in file_results if r.already_exists]),
            "execution_time_s": upload_time_total,
        },
    )

    return UploadResponse(
        session_id=session_id,
        files=[
            FileUploadResult(filename=r.filename, already_exists=r.already_exists)
            for r in file_results
        ],
    )
