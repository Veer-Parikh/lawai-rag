# Bharat Law Agent Instructions

## Mission

Bharat Law is becoming an evidence-first India Criminal-Law Change Monitor.
Prioritize accurate change tracking, source provenance, effective dates, and
reviewable citations over fluent but unsupported answers.

## Product boundary

- First jurisdiction: India.
- First corpus: BNS, BNSS, BSA, and the historical IPC, CrPC, and Indian
  Evidence Act material needed for cross-references.
- Include directly related central commencement, amendment, and rules
  instruments from 2023 onward.
- State law, broad all-India legal coverage, case prediction, and personalized
  legal advice are out of scope unless explicitly approved.

## Source and safety rules

- Treat Gazette or another primary legal instrument as controlling legal
  effect.
- Treat India Code as the authoritative consolidated-text representation.
- Treat MHA, PIB, and other explanatory publications as context, not as the
  legal source when a primary instrument is available.
- Preserve source URL, publisher, retrieval time, document hash, dates, and
  page/section anchors.
- Never present a provision as current without checking its effective status
  for the relevant date and jurisdiction.
- If evidence conflicts, is stale, or lacks the required anchor, expose the
  gap and abstain or ask for clarification.
- Do not imply that general information is legal advice; show the product
  disclaimer for public-facing explanations.
- Do not log raw user queries or uploaded legal documents by default.

## Engineering rules

- Read the relevant existing code and tests before changing behavior.
- Follow existing Python typing, logging, error, and configuration patterns.
- Keep retrieval providers behind interfaces so Pinecone can coexist with or
  be replaced by BM25 and HNSW implementations.
- Add focused tests for every behavior change and run the smallest relevant
  test command plus the full project suite before declaring completion.
- Do not silently swallow ingestion, retrieval, citation, or observability
  failures; return an explicit safe error or abstention.
- Record durable architecture choices in `skills/decisions.md`.
- Keep changes surgical and do not rewrite unrelated untracked work.

## Required validation

Before merging a feature, verify:

1. provenance and temporal metadata survive ingestion through answer output;
2. citations point to source anchors rather than only vector IDs;
3. evaluation cases cover both answerable and unanswerable questions;
4. traces include a request ID and the retrieval stages used;
5. frontend and backend contracts remain typed and backward-compatible.

## Useful commands

```powershell
pytest tests\ -v
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
python -m ingestion.run_ingestion --path data\ --doc-type all --dry-run
```
