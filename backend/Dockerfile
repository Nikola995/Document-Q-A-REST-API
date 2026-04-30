FROM python:3.12-slim

# System deps:
# - libgl1, libglib2.0-0: required by EasyOCR/OpenCV
# - libgomp1: required by FAISS
# - curl: useful for health checks during development
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy and install dependencies first (layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt


# Pre-download embedding model (used for chunk indexing and retrieval)
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

# Pre-download DistilBERT (used for QA inference)
RUN python -c "\
    from transformers import DistilBertTokenizer, DistilBertForQuestionAnswering; \
    DistilBertTokenizer.from_pretrained('distilbert-base-cased-distilled-squad'); \
    DistilBertForQuestionAnswering.from_pretrained('distilbert-base-cased-distilled-squad')"


COPY . .

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
