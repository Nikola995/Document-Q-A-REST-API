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


# Pre-download models — change only when model names change
ARG EMBEDDING_MODEL
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('${EMBEDDING_MODEL}')"

ARG QA_MODEL
RUN python -c "from transformers import DistilBertTokenizer; DistilBertTokenizer.from_pretrained('${QA_MODEL}')"
RUN python -c "from transformers import DistilBertForQuestionAnswering; DistilBertForQuestionAnswering.from_pretrained('${QA_MODEL}')"

ARG NER_MODEL
RUN python -c "from gliner2 import GLiNER2; GLiNER2.from_pretrained('${NER_MODEL}')"


COPY . .

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
