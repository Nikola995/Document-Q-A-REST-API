from fastapi import APIRouter, HTTPException
from app.models.schemas import QuestionRequest, AnswerResponse, ErrorResponse

router = APIRouter()


@router.post(
    "/ask",
    response_model=AnswerResponse,
    responses={404: {"model": ErrorResponse}, 502: {"model": ErrorResponse}},
    summary="Ask a question about a document",
    description="Given a document_id from /upload, retrieves the most relevant chunks via FAISS and generates an answer using the configured LLM.",
)
async def ask_question(body: QuestionRequest):
    answer = ""

    return AnswerResponse(
        document_id=body.document_id,
        question=body.question,
        answer=answer,
    )
