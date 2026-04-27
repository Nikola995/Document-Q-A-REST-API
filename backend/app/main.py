from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import documents, qa
from app.core.config import settings

app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    description="Document Q&A API — upload documents and ask questions about their content.",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(documents.router, prefix="/api/v1", tags=["documents"])
app.include_router(qa.router, prefix="/api/v1", tags=["qa"])


@app.get("/health", tags=["health"])
async def health_check():
    return {"status": "ok", "version": settings.VERSION}
