# Document Q&A API

A REST API that accepts document uploads (PDF / images) and answers questions about their content using semantic search + an LLM.

## Architecture

```
POST /api/v1/upload  →  extract text
POST /api/v1/ask     →  embed question
```

**Stack:** FastAPI
