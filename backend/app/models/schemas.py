from pydantic import BaseModel, Field
from datetime import datetime

# --- Upload ---


class FileUploadResult(BaseModel):
    filename: str
    already_exists: bool


class UploadResponse(BaseModel):
    session_id: str = Field(..., description="UUID identifying this session")
    files: list[FileUploadResult]


# --- Q&A ---


class QuestionRequest(BaseModel):
    session_id: str = Field(..., description="Session UUID returned from /upload")
    question: str = Field(..., min_length=3, max_length=1000)


class ContextChunk(BaseModel):
    filename: str
    text: str
    score: float = Field(..., description="Cosine similarity score")


class Entity(BaseModel):
    text: str
    label: str


class AnswerResponse(BaseModel):
    session_id: str
    question: str
    context_chunk: ContextChunk
    answer: str
    answer_entities: list[Entity] = []


# --- Error ---


class ErrorResponse(BaseModel):
    detail: str
