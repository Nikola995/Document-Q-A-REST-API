# Document Q&A API

A REST API that accepts document uploads (PDF / images) and answers questions about their content using a locally hosted QA model.

## Architecture

```
POST /api/v1/upload  →  extract text (PyMuPDF / EasyOCR)  →  store as text
POST /api/v1/ask     →  load document text  →  DistilBERT QA  →  answer
```

**Stack:** FastAPI · PyMuPDF · EasyOCR · DistilBERT · Redis · Streamlit · Docker

## Model

Question answering is handled locally by [`distilbert-base-cased-distilled-squad`](https://huggingface.co/distilbert-base-cased-distilled-squad) (66M parameters), loaded via Hugging Face `transformers` and run on CPU. DistilBERT is well within CPU inference capability and the official `transformers` examples for this model do not require GPU. The model is initialized once at application startup via FastAPI's lifespan and held in `app.state` for the lifetime of the process.

> **Note on GPU support:** Device-aware inference (CUDA) was explored but not included in this iteration due to a tensor device placement issue with the current `transformers` version when used outside of the `pipeline` abstraction. GPU support is noted as a future improvement.