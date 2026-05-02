# Document Q&A API

A REST API that accepts document uploads (PDF or images) and answers natural language questions about their content using a locally hosted QA model, session-based semantic search across multiple documents, and a Redis caching layer.

---

## Approach

The goal was to build a functional, well-structured AI backend that demonstrates awareness of production concerns (caching, observability, resource lifecycle) while avoiding overengineering for the assignment scope.

### Models and tools chosen

**DistilBERT (`distilbert-base-cased-distilled-squad`)** was chosen for question answering. At 66M parameters it runs comfortably on CPU, which the official Hugging Face examples reflect — no GPU is required. It is initialized once at application startup via FastAPI's lifespan context manager and held in `app.state` for the lifetime of the process, avoiding repeated loading overhead.

> **Note on GPU:** CUDA inference was explored but excluded due to a tensor device placement issue with the current `transformers` version when used outside of the `pipeline` abstraction. GPU support is noted as a future improvement.

**GLiNER (`gliner-community/gliner_medium-v2.5`)** was chosen for named entity recognition. It supports zero-shot NER with arbitrary entity labels defined at inference time, making it flexible without requiring fine-tuning. NER is run post-inference on the generated answer, not at indexing time — this avoids processing chunks that are never retrieved.

**sentence-transformers (`all-MiniLM-L6-v2`)** is used to embed document chunks for semantic retrieval via FAISS. The model is lightweight and produces strong retrieval quality for general-purpose text.

**FAISS** handles vector similarity search. It was chosen over managed vector store solutions (Pinecone, Chroma, etc.) deliberately — the manual metadata mapping it requires adds minimal complexity while keeping the stack self-contained and avoiding unnecessary external dependencies for this scope.

**FastAPI** was chosen for its native async support, automatic OpenAPI documentation, and idiomatic patterns for resource lifecycle management via lifespan.

**Redis** provides two caching layers: document-level deduplication (skipping re-indexing of identical documents detected via content hash) and query embedding caching (avoiding redundant embedding computation for repeated questions). The async `redis.asyncio` client is used throughout to avoid blocking FastAPI's event loop.

**Streamlit** provides a lightweight frontend for demonstrating the system end to end. It is intentionally minimal — see Future Considerations for the migration path to a production frontend.

**Loguru** is used for structured JSON logging (`serialize=True`), capturing per-file extraction times, inference latency, cache hit/miss events with associated metadata, retrieval scores, and document processing events. A dual-handler setup outputs human-readable logs to stderr for development and machine-readable JSON to a persistent log file for monitoring ingestion. `enqueue=True` ensures file writes are non-blocking and async-safe within FastAPI's event loop.

**pytest** with `pytest-asyncio` is used for the test suite. Tests cover endpoint validation, cache key correctness, chunking pipeline behavior, and text extraction on real invoice documents from the included test fixtures. `httpx.AsyncClient` with `ASGITransport` is used to test FastAPI routes without a running server.

### Session-based retrieval

Documents are organized into sessions — a `session_id` is generated on first upload and returned to the client, who includes it in subsequent uploads and all `/ask` requests. Each session maintains a single shared FAISS index across all its documents, enabling the system to act as a knowledge base over a document collection rather than querying individual files in isolation. Chunk metadata is stored alongside the index in a two-tier structure: a chunk-level store that preserves FAISS index order (mapping vector positions to chunk text and document ID) and a document-level store (mapping document IDs to filenames), keeping the two concerns cleanly separated. File extraction across a multi-file upload is performed concurrently via `asyncio.gather`, while indexing into the shared session index remains sequential to prevent write collisions.

---

## Architecture

```
POST /api/v1/upload
  → validate file types
  → generate session_id (first upload) or validate existing session_id
  → concurrent per-file processing (asyncio.gather):
      → save raw file to disk
      → extract text (PyMuPDF / EasyOCR)
      → check Redis cache (document deduplication by session-scoped content hash)
      → cache hit  → skip indexing, return existing document
      → cache miss → mark for indexing, cache document reference
  → sequential indexing into shared session FAISS index:
      → chunk text (semantic chunking via chonkie)
      → embed chunks (sentence-transformers)
      → append to session FAISS index + chunk metadata store + document metadata store
  → return session_id + per-file results

POST /api/v1/ask
  → validate session exists
  → check Redis cache (query embedding)
  → cache hit  → skip embedding step
  → cache miss → embed question (sentence-transformers) → cache embedding
  → FAISS similarity search across session index → retrieve top-k chunks
  → resolve chunk → document filename via two-tier metadata store
  → DistilBERT QA inference on top chunk
  → GLiNER NER on generated answer
  → return answer + context chunk (with filename + score) + answer entities
```

**Stack:** FastAPI · PyMuPDF · EasyOCR · chonkie · sentence-transformers · FAISS · DistilBERT · GLiNER · Redis · Loguru · Streamlit · Docker

---

## Project Structure

```
.
├── backend/
│   ├── app/
│   │   ├── main.py               # FastAPI app, middleware, lifespan
│   │   ├── core/config.py        # Settings via pydantic-settings
│   │   ├── models/
│   │   │   ├── schemas.py        # API request / response models (Pydantic)
│   │   │   └── internal.py       # Internal data models (dataclasses)
│   │   ├── api/routes/
│   │   │   ├── documents.py      # POST /upload
│   │   │   └── qa.py             # POST /ask
│   │   └── services/
│   │       ├── extraction.py     # PDF + OCR text extraction
│   │       ├── rag.py            # Chunking, FAISS indexing & retrieval
│   │       ├── llm.py            # DistilBERT QA inference
│   │       ├── ner.py            # GLiNER named entity recognition
│   │       └── cache.py          # Redis caching layer
│   ├── tests/
│   │   ├── conftest.py           # Shared pytest fixtures
│   │   ├── test_endpoints.py     # Route validation tests
│   │   ├── test_cache.py         # Cache key correctness tests
│   │   ├── test_rag.py           # Chunking and session existence tests
│   │   └── test_extraction.py    # PDF and OCR extraction tests
│   ├── requirements.txt
│   ├── .env.example
│   └── Dockerfile
├── frontend/
│   ├── app.py                    # Streamlit UI
│   ├── requirements.txt
│   └── Dockerfile
├── data/
│   └── invoices/
│       ├── invoice-pdfs/         # 30 sample PDF invoices
│       └── invoice-images/       # 30 sample scanned invoice images
├── docker-compose.yml
├── .gitignore
└── README.md
```

---

## Setup

### Requirements

- Python 3.12
- Docker and Docker Compose

### Docker (recommended)

The simplest way to run the full stack. Models are pre-downloaded into the image at build time, so there is no cold-start download on first request.

```bash
# 1. Copy environment config
cp .env.example .env

# 2. Build and start all services
docker compose up --build

# 3. Access
#   API docs:  http://localhost:8000/docs
#   UI:        http://localhost:8501
```

To run only the backend and Redis (without the Streamlit frontend):

```bash
docker compose up --build backend
```

> **Note on standalone backend builds:** `docker compose up --build` is the recommended build path as it automatically passes model names from `.env` as build arguments to the Dockerfile. If building the backend image directly with `docker build`, model names must be passed explicitly via `--build-arg`:
> ```bash
> docker build \
>   --build-arg QA_MODEL=distilbert-base-cased-distilled-squad \
>   --build-arg EMBEDDING_MODEL=all-MiniLM-L6-v2 \
>   --build-arg NER_MODEL=fastino/gliner2-base-v1 \
>   ./backend
> ```
> Values should match those in `/backend/.env` to ensure the pre-downloaded models match the runtime configuration.

### Manual Installation

```bash
# 1. Create and activate a virtual environment (uv recommended)
cd backend
uv venv
source .venv/bin/activate

# 2. Install dependencies
uv pip install -r requirements.txt

# 3. Start Redis (required — run in a separate terminal)
docker compose up redis

# 4. Copy and configure environment
cp .env.example .env
# For local development, ensure REDIS_URL=redis://localhost:6379 in .env
# Model names are specified in .env — update these before running if using different models

# 5. Start the backend
uvicorn app.main:app --port 8000 --reload

# 6. (Optional) Start the frontend in a separate terminal
cd ../frontend
uv venv
source .venv/bin/activate
uv pip install -r requirements.txt
streamlit run app.py
```

---

## API

### `POST /api/v1/upload`

Upload one or more PDFs or images for text extraction and indexing into a session.

**Accepted formats:** `application/pdf`, `image/png`, `image/jpeg`, `image/tiff`, `image/webp`

**Body:** `multipart/form-data` with one or more `files` fields and an optional `session_id` form field. Omit `session_id` on first upload — a new session will be created and returned.

**Response:**
```json
{
  "session_id": "3f7a1c2d-...",
  "files": [
    { "filename": "invoice.pdf", "already_exists": false },
    { "filename": "contract.pdf", "already_exists": true }
  ]
}
```

`already_exists: true` is returned per file when an identical document has already been indexed in the current session, detected via a SHA-256 hash of the extracted text. No re-indexing is performed for duplicate files. All files are validated before any processing begins — an unsupported file type in a batch will reject the entire request.

---

### `POST /api/v1/ask`

Ask a natural language question across all documents in a session.

**Body:**
```json
{
  "session_id": "3f7a1c2d-...",
  "question": "What are the payment terms?"
}
```

**Response:**
```json
{
  "document_id": "3f7a1c2d-...",
  "question": "Who is the client that sent an invoice in 2024?",
  "answer": "In 2024, John Smith sent...",
  "source_chunk": {
    "text": "...the retrieved passage used as context...",
    "score": 0.91,
    "filename": "invoice.pdf"
  },
  "entities": [
    { "text": "John Smith", "label": "person"},
    { "text": "2024", "label": "date"}
  ]
}
```

`score` reflects the cosine similarity between the question embedding and the retrieved chunk across the session index. `filename` identifies which document in the session the context chunk originated from. `answer_entities` are extracted from the answer text by GLiNER using a fixed label set defined in config.

---

## Test Data

Sample documents used for testing are sourced from the following publicly available datasets:

- **Invoice PDFs:** [Sample PDF Invoices](https://github.com/femstac/Sample-Pdf-invoices)
- **Invoice Images:** [High Quality Invoice Images for OCR](https://www.kaggle.com/datasets/osamahosamabdellatif/high-quality-invoice-images-for-ocr) via Kaggle — licensed under [Open Data Commons Database Contents License (DbCL) v1.0](https://opendatacommons.org/licenses/dbcl/1-0/), which permits free use, reproduction and distribution including for commercial purposes.

A subset of 30 samples from each dataset is included in `/data/invoices/`.

---

## Running Tests

```bash
cd backend
pytest tests/ -v
```

Tests use `httpx.AsyncClient` with `ASGITransport` — no running server required.

---

## Demo

See [demo/README.md](demo/README.md) for a visual walkthrough of the application.

---

## Future Considerations

The following improvements were identified during development but excluded from this iteration as beyond the current scope.

**Advanced retrieval techniques**
The current retrieval uses dense vector search (FAISS) against a single top-k chunk. Natural next steps include hybrid search — combining dense retrieval with BM25 sparse keyword matching, merged via Reciprocal Rank Fusion — and re-ranking, a two-stage pipeline where a cross-encoder re-scores a larger candidate set before passing top-k to the QA model. Both significantly improve retrieval quality on diverse document types.

**Document-type-aware NER**
The current NER implementation uses a fixed label set defined in config, applied uniformly across all document types. A natural extension would be document-type-aware label selection — where the document type is either specified manually by the client at upload time, or determined automatically as part of the pipeline via a document classification step (with the result stored as part of the document metadata). The backend would then select the appropriate label set from config accordingly. For more flexible use cases, labels could also be made configurable per `/ask` request, shifting label selection to the caller entirely. Both approaches are straightforward extensions of the current architecture without requiring model changes, as GLiNER2 is zero-shot and generalizes across arbitrary label types.

**User management and authentication**
The current session model provides document isolation at the session level but has no concept of identity — any client who knows a `session_id` can upload to or query it. The natural progression is user accounts with JWT authentication, where sessions are scoped to authenticated users rather than bare UUIDs. This would make sessions persistent across devices and requests, enable per-user storage quotas, and provide a clear ownership model for document collections. It also introduces security considerations that are currently deferred: session IDs exposed as client-visible identifiers become access tokens in effect, and without authentication there is no mechanism to prevent unauthorized access or session enumeration. Rate limiting (see below) partially mitigates abuse, but authentication is the correct long-term solution.

**GPU inference**
Device-aware inference was explored but excluded due to a tensor device placement issue with the current `transformers` version outside of the `pipeline` abstraction. For larger models this would be a prerequisite rather than an optimization.

**Monitoring and performance profiling**
Structured JSON logging is already implemented via Loguru, capturing inference latency, cache hit/miss events with metadata, retrieval scores, and per-file processing times. The natural next step is ingesting these logs into a proper observability stack such as Grafana + Loki, enabling dashboards and alerting without modifying application code. A partial solution such as a metrics endpoint or a Streamlit monitoring view was intentionally avoided in favour of implementing the logging layer correctly and leaving the aggregation layer for when it is genuinely needed. The existing latency data also provides a foundation for profiling model inference and API throughput, identifying bottlenecks across the extraction, embedding, retrieval, and inference stages — a natural extension once a monitoring stack is in place.

**Rate limiting and input sanitization**
`slowapi` (a FastAPI wrapper around `limits`) would allow per-endpoint rate limiting backed by the existing Redis instance, with more aggressive limits on `/upload` (extraction and indexing are expensive) than `/ask`. Input sanitization — validating and cleaning extracted text before it reaches the embedding and QA models — is a complementary concern. Both are standard production hardening measures that were excluded as they are not representative of the current development and testing environment, but would be required before any public-facing deployment.

**Microservice architecture**
The three models (DistilBERT, GLiNER, sentence-transformers) currently share a single FastAPI process. If workloads became imbalanced — for example the embedding model receiving significantly higher traffic than the NER model — each model would be a natural candidate for extraction into an independent service with its own scaling policy. For the current scope this would be overengineering.

**Streaming responses**
For longer documents or heavier models, streaming intermediate pipeline status (extraction complete, indexing complete, etc.) would improve perceived responsiveness. This requires Server-Sent Events on the backend and polling or SSE support on the frontend, and is only justified when processing times exceed several seconds consistently.

**Testing in CI/CD**
The test suite currently runs locally. Natural next steps are a Docker multi-stage build that runs tests as part of `docker build` (failing the build on test failure) and a GitHub Actions workflow that runs tests on every push. Rate limiting and input sanitization would also expand the test surface meaningfully. These were omitted as the overhead is not justified at the current project scale.

**Production frontend**
Streamlit is well suited for demos and internal tooling but is not intended for production user-facing applications. A React frontend would provide proper component architecture, full CSS control, hover interactions, and native SSE support for streaming responses.

**Cloud deployment**
Moving to production would involve containerized deployment to a cloud provider (AWS ECS, GCP Cloud Run, etc.), a managed Redis instance (ElastiCache, Memorystore), persistent volume storage for FAISS indexes, and a reverse proxy (nginx) in front of the FastAPI backend for TLS termination, request buffering, and load balancing across multiple backend instances.
