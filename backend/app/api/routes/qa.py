from fastapi import APIRouter, Request, HTTPException
from app.models.schemas import QuestionRequest, AnswerResponse, ErrorResponse
from pathlib import Path
from app.services import llm
from app.core.config import settings

router = APIRouter()


@router.post(
    "/ask",
    response_model=AnswerResponse,
    responses={404: {"model": ErrorResponse}, 502: {"model": ErrorResponse}},
    summary="Ask a question about a document",
    description="Given a document_id from /upload, retrieves the most relevant chunks via FAISS and generates an answer using the configured LLM.",
)
async def ask_question(body: QuestionRequest, request: Request):
    # Retrieve document from its uuid as context to the question
    try:
        # loading extracted text as single index (TODO: replace with indexing when implemented)
        doc_path = Path(settings.INDEX_DIR) / f"{body.document_id}.txt"
        if not doc_path.exists():
            raise FileNotFoundError
        context = doc_path.read_text(encoding="utf-8")
    except FileNotFoundError:
        raise HTTPException(
            status_code=404,
            detail=f"No index found for document_id '{body.document_id}'. Did you upload it first?",
        )

    # Generate answer via LLM
    try:
        answer = await llm.generate_answer(
            question=body.question, context=context, request=request
        )
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"LLM call failed: {e}")

    return AnswerResponse(
        document_id=body.document_id,
        question=body.question,
        answer=answer,
    )
