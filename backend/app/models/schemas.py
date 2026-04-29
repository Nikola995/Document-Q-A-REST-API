from pydantic import BaseModel, Field
from datetime import datetime

# --- Upload ---


class UploadResponse(BaseModel):
    document_id: str = Field(..., description="UUID identifying this document session")
    filename: str
    num_chunks: int
    created_at: datetime


# --- Q&A ---


class QuestionRequest(BaseModel):
    document_id: str = Field(..., description="UUID returned from /upload")
    question: str = Field(..., min_length=3, max_length=1000)


class SourceChunk(BaseModel):
    text: str
    score: float = Field(..., description="Cosine similarity score")


class AnswerResponse(BaseModel):
    document_id: str
    question: str
    answer: str


# --- Error ---


class ErrorResponse(BaseModel):
    detail: str
