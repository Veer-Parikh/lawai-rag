# Bharat Law 🏛️

Evidence-first India Criminal-Law Change Monitor.

**Current stack:** FastAPI · Gemini 2.5 Flash · BGE-base-en · BAAI/bge-reranker-base · Pinecone Serverless  
**Target architecture:** Next.js · BM25 + HNSW + RRF · versioned legal evidence · OpenTelemetry-compatible observability

The first release is intentionally bounded to BNS, BNSS, BSA, their historical
IPC/CrPC/Indian Evidence Act crosswalks, and directly related central
commencement/amendment instruments. It prioritizes source provenance,
effective-date correctness, and visible abstention over unsupported fluency.

---

## Quick Start

### 1. Install dependencies
```bash
python -m pip install --upgrade pip setuptools wheel
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
| `POST` | `/auth/demo` | One-click demo session (researcher role) |
| `POST` | `/auth/admin` | Admin session from `ADMIN_PASSWORD` |
| `GET` | `/auth/me` | Current session |
| `POST` | `/feedback` | Rate an answer (signed in; stores a query hash only) |
| `GET` | `/documents` | List document versions in the legal catalog |
| `POST` | `/admin/ingest` | Ingest a file or folder inside `DATA_DIR` (admin) |
| `GET` | `/admin/ingest/status/{job_id}` | Job status with live progress (admin) |
| `GET` | `/admin/ingest/jobs` | Jobs since API start (admin) |
| `GET` | `/admin/corpus` | Files, provenance, ingestion status, history (admin) |
| `POST` | `/admin/corpus/upload` | Upload a PDF into a corpus folder (admin) |
| `GET` | `/admin/analytics?days=14` | Usage, quality and latency analytics (admin) |
| `GET` | `/admin/index` | Pinecone vector counts per namespace (admin) |

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
│   ├── eval_logger.py     # Compatibility JSONL metrics log
│   ├── eval_metrics.py    # Dependency-free retrieval metrics
│   └── tracing.py         # OpenTelemetry-compatible local trace contract
├── legal/
│   └── evidence.py        # Version-aware status and source authority model
├── evals/
│   ├── datasets/          # Versioned evaluation cases
│   └── run_eval.py        # Offline evaluation runner
├── frontend-next/         # Next.js lawyer research workspace
├── skills/                # Agent operating contracts
│   ├── copilot-instructions.md
│   ├── decisions.md
│   ├── frontend.md
│   ├── backend.md
│   └── rag.md
├── docs/superpowers/specs/ # Approved product design
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

## Web app, sign-in and admin console

```bash
cd frontend-next && npm install && npm run dev   # http://localhost:3000
```

- **Demo login:** "Continue as demo researcher" on `/login` opens the research
  workspace. Turn it off with `DEMO_LOGIN_ENABLED=false`.
- **Admin login:** set `ADMIN_PASSWORD` (and `AUTH_SECRET`, so sessions survive
  restarts) in `.env`, then sign in as administrator. Admins get
  `/admin` (usage, abstention rate, latency, feedback, API traffic) and
  `/admin/corpus` (upload PDFs, see which files are searchable, run or test-run
  ingestion with live progress, see sources that failed to download).

Auth is demo-grade: signed tokens, no user database. See `skills/decisions.md`.

## Admin API

Trigger ingestion via HTTP with an admin bearer token, or the `X-Admin-Key` header:

```bash
curl -X POST http://localhost:8000/admin/ingest \
  -H "X-Admin-Key: your_admin_key" \
  -H "Content-Type: application/json" \
  -d '{"target_path": "judgments/supreme_court", "doc_type": "judgment"}'
```

---

## Disclaimer

Bharat Law is for informational purposes only and does not constitute legal advice.
Always consult a qualified lawyer for your specific legal situation.
