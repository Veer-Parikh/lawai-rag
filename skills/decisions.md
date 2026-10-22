# Architecture and Product Decisions

This file is append-only. Add a dated entry when a decision changes product
scope, source authority, data contracts, retrieval, observability, or
deployment. Record the reason and the cost if the decision is wrong.

## 2026-10-19 — Product boundary

- **Decision:** Build an India Criminal-Law Change Monitor first, focused on
  BNS, BNSS, BSA, and historical IPC/CrPC/Indian Evidence Act crosswalks.
- **Reason:** The original problem is tracking continuously released legal
  changes; a bounded criminal-law corpus makes provenance, versioning, and
  evaluation measurable.
- **Alternatives rejected:** A general Indian-law chatbot would create an
  unbounded corpus and make freshness and correctness difficult to establish.
- **Cost if wrong:** Other legal domains will require a later scope expansion
  and additional source/evaluation work.

## 2026-10-19 — Source authority

- **Decision:** Gazette or another primary legal instrument controls legal
  effect; India Code controls consolidated text; MHA/PIB is explanatory.
- **Reason:** Users need to distinguish legally operative instruments from
  summaries and consolidated representations.
- **Cost if wrong:** Conflicting official sources could produce an incorrect
  effective-date or amendment answer.

## 2026-10-19 — Version and commencement model

- **Decision:** Store provision-level effective intervals, exceptions,
  territories, conditions, and unknown status explicitly.
- **Reason:** A single `current=true` flag cannot represent staged or partial
  commencement and would make historical questions unsafe.
- **Cost if wrong:** Temporal filtering and change timelines may need a schema
  migration.

## 2026-10-19 — Frontend migration

- **Decision:** Replace Streamlit with a professional TypeScript Next.js
  frontend while retaining the FastAPI service boundary initially.
- **Reason:** Next.js provides a maintainable product UI and typed client
  contracts without coupling the migration to a backend rewrite.
- **Cost if wrong:** The project will carry a temporary dual-frontend period
  while the Streamlit UI is retired.

## 2026-10-19 — Retrieval direction

- **Decision:** Evaluate hybrid BM25 and HNSW retrieval, fuse candidates with
  reciprocal rank fusion, then rerank and expand legal relationships.
- **Reason:** Exact section and citation terms benefit from lexical search,
  while natural-language questions benefit from dense retrieval.
- **Cost if wrong:** Maintaining multiple retrieval paths increases tuning and
  observability complexity.

## 2026-10-19 — Evaluation and observability

- **Decision:** Treat versioned offline evals, citation/temporal metrics, and
  correlated traces as release requirements.
- **Reason:** Legal RAG quality cannot be established from answer fluency or
  one append-only JSONL log.
- **Cost if wrong:** Initial implementation takes longer, but omitting this
  would make regressions and unsupported answers difficult to detect.

## 2026-10-21 — Document comparison as a first-class workflow

- **Decision:** Expose document-version summaries and comparison results as
  typed API contracts, then surface the change summary in the Next.js workspace.
- **Reason:** The product's distinctive value is tracking what changed, not
  only answering a question against a vector index.
- **Cost if wrong:** The initial catalog is in-memory and must move behind
  durable storage before multi-user production use.

## 2026-10-21 — Python 3.12 build compatibility

- **Decision:** Require modern `setuptools` and `wheel`, and constrain NumPy
  to `>=1.26.4,<2.0` for the supported Python 3.12 environment.
- **Reason:** NumPy 1.24.x has no compatible Python 3.12 wheel and falls back
  to a legacy source build that imports the removed standard-library
  `distutils` module. The explicit build-tool bootstrap and compatible NumPy
  range make dependency installation reproducible.
- **Cost if wrong:** Older Python environments may need a separate
  compatibility requirements file rather than using the main requirements.

## 2026-10-21 — Evidence-backed citation audit

- **Decision:** Treat citations as verified only when structured or inline
  references map to retrieved chunk metadata; expose the mapping, authority
  URL, and page anchor in the `/chat` response.
- **Reason:** A model-generated section or case reference is not evidence by
  itself. Mapping references back to retrieved chunks makes unsupported
  citations visible to clients and allows the UI to link to authoritative
  material.
- **Cost if wrong:** Some legitimate citations absent from the retrieved
  context will be flagged until retrieval coverage or source metadata improves.

## 2026-10-21 — Demo-grade authentication with two roles

- **Decision:** Add a one-click demo login (role `researcher`) and a password
  admin login (role `admin`, `ADMIN_PASSWORD`), issuing stdlib HMAC-signed
  bearer tokens. Admin routes accept the token or the legacy `X-Admin-Key`.
- **Reason:** The pilot needs a separated admin surface (analytics, corpus,
  ingestion) without introducing an identity provider or a user database yet.
- **Alternatives rejected:** NextAuth/OAuth (premature for a single-team pilot);
  frontend-only gating (would leave admin APIs open).
- **Cost if wrong:** No per-user identity, revocation, or audit trail. Replace
  with a real identity provider before multi-tenant or public use; `/chat` is
  still callable without a token.

## 2026-10-21 — Ingestion ledger and deterministic chunk IDs

- **Decision:** Record every file ingestion attempt in
  `data/ingestion_ledger.sqlite3`, and derive chunk IDs from the file's path
  inside `DATA_DIR` plus chunk position (UUIDv5) instead of random UUIDs.
- **Reason:** Administrators need to see what is searchable, and one-click
  re-ingestion must overwrite vectors rather than duplicate them.
- **Cost if wrong:** If a re-ingested file now yields fewer chunks, the extra
  old vectors remain until the namespace is cleaned. Moving a file changes its
  IDs. Vectors ingested before this change still have random IDs.

## 2026-10-21 — Research feedback log

- **Decision:** Store answer feedback (`wrong_source`, `stale_status`,
  `missing_authority`, `poor_explanation`, `helpful`) keyed by query hash in
  `observability/feedback.jsonl`; aggregate it in admin analytics.
- **Reason:** The frontend skill requires these feedback categories and they
  are the cheapest legal-quality signal available before human review.
- **Cost if wrong:** JSONL does not scale past a pilot; move to the same store
  as traces when observability is consolidated.
