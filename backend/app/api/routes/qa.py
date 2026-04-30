from fastapi import APIRouter, Request, HTTPException
from app.models.schemas import QuestionRequest, AnswerResponse, ErrorResponse
from app.services import llm, rag

router = APIRouter()


@router.post(
    "/ask",
    response_model=AnswerResponse,
    responses={404: {"model": ErrorResponse}, 502: {"model": ErrorResponse}},
    summary="Ask a question about a document",
    description="Given a document_id from /upload, retrieves the most relevant chunks via FAISS and generates an answer using the configured LLM.",
)
async def ask_question(body: QuestionRequest, request: Request):
    # Retrieve top-k chunks from FAISS
    try:
        context = await rag.retrieve_chunks(
            document_id=body.document_id, question=body.question, request=request
        )
    except FileNotFoundError:
        raise HTTPException(
            status_code=404,
            detail=f"No index found for document_id '{body.document_id}'. Did you upload it first?",
        )

    if not context:
        raise HTTPException(
            status_code=404, detail="No relevant content found for this question."
        )

    # Generate answer via LLM
    try:
        answer = await llm.generate_answer(
            question=body.question, context=context.text, request=request
        )
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"LLM call failed: {e}")

    return AnswerResponse(
        document_id=body.document_id,
        question=body.question,
        context=context,
        answer=answer,
    )
