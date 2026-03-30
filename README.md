# Bharat Law 🏛️

Production-grade RAG-based legal chatbot for Indian law.

**Stack:** FastAPI · Gemini 2.5 Flash · BGE-base-en · BAAI/bge-reranker-base · Pinecone Serverless

---

## Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
# Also install Tesseract binary for OCR (optional, for scanned PDFs):
# Windows: https://github.com/UB-Mannheim/tesseract/wiki
```

### 2. Set environment variables
```bash
cp .env.example .env
# Fill in PINECONE_API_KEY, GEMINI_API_KEY, ADMIN_API_KEY
```

### 3. Ingest your documents (one-time)
```bash
# Full ingest
python -m ingestion.run_ingestion --path data/ --doc-type all

# Dry run first (no upload)
python -m ingestion.run_ingestion --path data/ --doc-type all --dry-run

# Specific folder
python -m ingestion.run_ingestion --path data/statutes/criminal --doc-type statute
```

### 4. Start the API
```bash
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```

API docs available at: [http://localhost:8000/docs](http://localhost:8000/docs)

---

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/health` | Liveness + dependency check |
| `POST` | `/chat` | Full RAG Q&A with Gemini |
| `GET` | `/search` | Raw retrieval (no LLM) |
| `POST` | `/admin/ingest` | Trigger ingestion job |
| `GET` | `/admin/ingest/status/{job_id}` | Check job status |

### Example `/chat` request
```json
POST /chat
{
  "query": "What is the punishment for murder under IPC?",
  "domain": "criminal",
  "top_k": 5,
  "session_id": "user-session-abc"
}
```

### Example `/search` request
```
GET /search?q=Section+302+IPC&domain=criminal&top_k=10
```

---

## Project Structure

```
lawai-rag/
├── config.py              # All config + env vars
├── requirements.txt
├── ingestion/             # One-time PDF → Pinecone pipeline
│   ├── pdf_loader.py      # PDF text + OCR fallback
│   ├── statute_chunker.py # Section-aware chunking
│   ├── judgment_chunker.py# Paragraph + headnote chunking
│   ├── mapping_parser.py  # IPC↔BNS cross-ref extraction
│   ├── metadata_builder.py# Pinecone metadata construction
│   ├── embedder.py        # BGE-base-en embeddings
│   ├── pinecone_uploader.py
│   └── run_ingestion.py   # CLI entry point
├── retrieval/             # Query-time retrieval
│   ├── domain_router.py   # 2-stage: keyword + LLM + LRU cache
│   ├── query_rewriter.py  # Follow-up query rewriting
│   ├── pinecone_retriever.py
│   ├── reranker.py        # BAAI/bge-reranker-base
│   ├── retrieval_quality_assessor.py  # HIGH/MEDIUM/LOW gate
│   └── context_builder.py # XML context + cross-ref append
├── llm/
│   ├── prompt_builder.py  # System prompt + citation rules
│   └── gemini_client.py   # Gemini 2.5 Flash + citation validation
├── api/
│   ├── main.py            # FastAPI app + model warm-up
│   ├── middleware.py      # CORS, rate limit, logging
│   ├── routers/
│   │   ├── chat.py        # POST /chat (full pipeline)
│   │   ├── search.py      # GET /search (retrieval only)
│   │   ├── health.py
│   │   └── admin.py       # Ingestion trigger
│   └── schemas/
├── observability/
│   └── eval_logger.py     # Append-only JSONL eval log
├── utils/
│   ├── logger.py
│   ├── text_cleaner.py
│   └── exceptions.py
└── tests/
    ├── unit/
    └── integration/
```

---

## Key Design Decisions

| Decision | Choice | Reason |
|---|---|---|
| Pinecone tier | Serverless | Pay-per-use, no pod provisioning |
| Embeddings | BGE-base-en (768d) | Strong EN legal retrieval, local |
| Reranker | BAAI/bge-reranker-base | Same family as embedder, better legal fit |
| Domain routing | 2-stage keyword + LLM | Fast path for clear queries, LLM for ambiguous |
| Confidence gate | 3-tier (HIGH/MEDIUM/LOW) | Blocks LLM calls on poor retrieval |
| Session store | In-memory dict | Simple, zero-dependency; migrate to Redis in v1.2 |
| Eval logging | JSONL file | Append-only, privacy-safe (no raw queries) |

---

## Running Tests

```bash
# Unit tests only (no API keys needed)
pytest tests/unit/ -v

# Integration tests (uses mocks, no real API calls)
pytest tests/integration/ -v

# All tests
pytest tests/ -v
```

---

## Admin API

Trigger ingestion via HTTP (requires `X-Admin-Key` header):

```bash
curl -X POST http://localhost:8000/admin/ingest \
  -H "X-Admin-Key: your_admin_key" \
  -H "Content-Type: application/json" \
  -d '{"target_path": "data/supreme_court_judgments/2025", "doc_type": "judgment"}'
```

---

## Disclaimer

Bharat Law is for informational purposes only and does not constitute legal advice.
Always consult a qualified lawyer for your specific legal situation.
