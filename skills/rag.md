# RAG Skill — Versioned Indian Criminal-Law Retrieval

## Corpus and provenance

The first corpus is BNS, BNSS, BSA, and the historical IPC, CrPC, and Indian
Evidence Act material needed for crosswalks, plus directly related central
commencement, amendment, and rules instruments from 2023 onward.

Source precedence:

1. Gazette or primary legal instrument for legal effect.
2. India Code for consolidated statutory text.
3. MHA/PIB and similar official explanations for context.
4. Curated judgments for interpretation, never as a replacement for primary
   statutory authority.

Each document/version and chunk must retain stable IDs, source URL, publisher,
retrieval time, hash, language, document type, extraction/OCR status, page or
section anchor, publication date, effective date, and legal status.

## Legal version model

Represent provision-level effective intervals and relationships explicitly:

- enacted, notified, in force, superseded, repealed, or unknown;
- commencement exceptions, conditions, and territorial scope;
- predecessor/successor and amendment-instrument links;
- IPC/CrPC/Indian Evidence Act to BNS/BNSS/BSA cross-references.

Never infer legal currency from embedding similarity, ingestion order, file
name, or a single boolean current flag.

## Retrieval pipeline

The target query path is:

1. normalize the query and extract act, section, date, and jurisdiction;
2. retrieve lexical candidates with BM25;
3. retrieve dense candidates from an HNSW-backed index;
4. fuse rankings with reciprocal rank fusion (RRF);
5. apply legal metadata and temporal filters;
6. rerank a bounded candidate set with a cross-encoder;
7. expand linked amendments, predecessors, successors, and commencement
   evidence;
8. build citation-preserving context;
9. answer only when the evidence gate passes.

Keep Pinecone behind a retrieval adapter while BM25 and HNSW are introduced.
The interface should expose candidate IDs, scores, source metadata, and
retrieval stage without leaking vendor-specific response objects.

## Evidence gate

An answer is supported only when required source anchors and temporal
constraints are satisfied. If evidence is missing, conflicting, stale, or
outside the requested jurisdiction/date:

- state what could not be established;
- show the relevant source conflict or missing evidence;
- abstain or ask a targeted clarification;
- never fill the gap with model memory.

Vector, reranker, or LLM scores are retrieval signals, not legal certainty.

## Evaluation

Maintain versioned, reviewed datasets for exact section lookup,
IPC/CrPC/Evidence Act crosswalks, commencement/effective-date questions,
amendment and supersession questions, date-specific “in force” questions,
source-priority questions, multilingual/transliterated queries if supported,
and unanswerable/conflicting cases.

Track at least:

- Recall@k, Precision@k, MRR, and nDCG;
- citation recall and citation entailment;
- temporal-filter accuracy and amendment-link accuracy;
- groundedness, completeness, refusal quality, and human legal-review score;
- latency, token usage, cost, ingestion freshness, and failure rates.

Every experiment records corpus snapshot, chunking version, embedding model,
BM25/HNSW/RRF settings, reranker, prompt version, and eval-set version.

## Observability

Trace ingestion and query stages separately. Include request/trace IDs,
document/version IDs and hashes, model/index/config versions, stage latency,
token counts, cost, errors, citation warnings, and abstention reasons.
Redact raw legal documents and user queries by default; make payload capture
explicit and controlled.

## Anti-patterns

- Using only dense retrieval for section-number or citation-heavy queries.
- Treating consolidated text as proof of commencement.
- Using unofficial aggregators as the primary authority when an official
  source exists.
- Returning a fluent answer with no source anchor.
- Measuring only answer quality while ignoring retrieval and temporal
  correctness.
- Swallowing parser, index, citation, or telemetry failures.
