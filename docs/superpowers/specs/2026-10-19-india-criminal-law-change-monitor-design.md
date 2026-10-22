# India Criminal-Law Change Monitor

## Intent

Bharat Law should help Indian legal professionals keep track of changes to
criminal law without having to repeatedly compare scattered PDFs, Gazette
notifications, and government pages by hand. A secondary public mode should
explain a retrieved provision in plain language, while clearly separating
general information from legal advice.

The product is not intended to be a general-purpose legal chatbot. Its
distinctive value is an evidence-first change timeline:

1. identify the authoritative document and jurisdiction;
2. preserve the document version and publication/effective dates;
3. link amendments, commencement notifications, and related provisions;
4. show what changed between versions;
5. answer questions using citations to the source passage and its status.

## Proposed first project statement

**Build a source-grounded India Criminal-Law Change Monitor that tells a
lawyer what changed in the BNS, BNSS, and BSA, when the change took effect,
which official source proves it, and how the affected provision relates to
the previous law.**

The public-facing experience uses the same evidence layer to explain a
document or provision in understandable language. It must not imply that the
system has assessed a user's individual legal position.

## Primary users and success

### Primary user

Indian lawyers and legal researchers who need to:

- check the current text of a criminal-law provision;
- locate the prior or substituted provision;
- verify commencement and amendment dates;
- understand the relationship between IPC/CrPC/Indian Evidence Act and
  BNS/BNSS/BSA;
- produce a source-backed research note quickly.

### Secondary user

Members of the public who have found an Indian criminal-law document and want
an understandable explanation of what it says, whether it is current, and
where it came from.

### Initial success criteria

- A lawyer can answer a section-change question with a source citation and
  effective-date evidence in under a few minutes.
- The system distinguishes enacted, notified, and in-force text rather than
  treating all retrieved text as current law.
- Retrieval evaluation demonstrates reliable section, date, and amendment
  linking before answer-generation quality is optimized.
- Unsupported or ambiguous questions receive a transparent abstention or
  clarification request.
- Every generated answer can be traced to retrieval, reranking, prompt,
  model, latency, and evaluation metadata without storing raw sensitive
  queries by default.

## Initial legal scope

The first release covers central criminal-law materials:

- Bharatiya Nyaya Sanhita, 2023 (BNS);
- Bharatiya Nagarik Suraksha Sanhita, 2023 (BNSS);
- Bharatiya Sakshya Adhiniyam, 2023 (BSA);
- the corresponding IPC, Code of Criminal Procedure, and Indian Evidence
  Act text needed for historical comparison;
- official commencement notifications, amendment notifications, rules, and
  directly related Ministry of Home Affairs material;
- a deliberately small, curated set of Supreme Court judgments used to
  explain interpretation, not to replace primary statutory sources.

State criminal laws, every High Court judgment, civil law, tax law, and
personalized legal advice are outside the first release.

## Data plan

### Source priority

1. **India Code** for the official current central Act text and section
   structure.
2. **e-Gazette** for enacted publications, commencement notifications,
   amendments, rules, and other legally operative notifications.
3. **Ministry of Home Affairs** for official publications and explanatory
   material related to the three new criminal codes.
4. **Supreme Court official judgment sources** for the curated interpretive
   corpus.
5. **Open datasets** such as the AWS Supreme Court judgments dataset may be
   used for research and bootstrapping, but not treated as the authority
   when an official source is available.

Source adapters must record URL, publisher, retrieval timestamp, document
hash, publication date, effective date when known, language, document type,
and extraction/OCR status. The ingestion pipeline must retain the original
file and a normalized text representation so extraction errors can be audited.

### Canonical document model

Each document/version should have stable identifiers for:

- act, rule, notification, or judgment;
- section/article/rule and subsection;
- source publisher and source URL;
- publication, assent, notification, and effective dates;
- status: enacted, notified, in force, superseded, repealed, or unknown;
- predecessor/successor relationships;
- amendment and commencement relationships;
- language and page/paragraph anchors;
- extraction confidence and provenance.

This model is more important than choosing a vector database. It prevents a
semantically similar but legally stale chunk from being presented as current.

## Retrieval and answer architecture

The existing Python backend can remain the first service boundary while the
retrieval core is hardened.

The target retrieval path is:

1. normalize the query and detect act, section, date, and jurisdiction
   constraints;
2. retrieve lexical candidates with BM25;
3. retrieve dense candidates from an HNSW-backed vector index;
4. fuse lexical and dense rankings with reciprocal rank fusion (RRF);
5. apply metadata and temporal filters;
6. rerank a bounded candidate set with a cross-encoder;
7. expand linked amendment, predecessor, successor, and commencement
   evidence;
8. build a citation-preserving context;
9. generate an answer only when evidence and source status meet the quality
   gate.

Pinecone may remain available during migration, but the interface should
separate retrieval contracts from the vendor so an HNSW-backed local or
managed index can be evaluated fairly. The implementation should not claim
that a vector score is legal confidence.

## Evaluation

Evaluation is a first-class feature, not a log file added after the chatbot.

### Offline datasets

Create versioned, reviewed evaluation sets for:

- exact section lookup;
- IPC/CrPC/Evidence Act to BNS/BNSS/BSA mapping;
- commencement and effective-date questions;
- amendment and supersession questions;
- temporal questions such as “what was in force on date X?”;
- source-priority and citation questions;
- multilingual or transliterated queries if supported;
- unanswerable and conflicting-source cases.

Each case should include the question, expected source document IDs,
expected section anchors, temporal constraints, acceptable answer facts, and
an abstention label when appropriate.

### Metrics

Track retrieval metrics such as Recall@k, Precision@k, MRR, nDCG, citation
recall, temporal-filter accuracy, and amendment-link accuracy. Track answer
metrics such as groundedness, citation entailment, completeness, refusal
quality, and human legal-review scores. Track operational metrics such as
latency by stage, token usage, cost, index health, ingestion freshness, and
failure rates.

Every retrieval experiment must record corpus snapshot, chunking version,
embedding model/version, BM25 settings, HNSW settings, RRF constant,
reranker, prompt version, and evaluation-set version.

## Observability

Replace the current privacy-limited JSONL-only approach with structured,
correlated traces and metrics while retaining a safe audit record.

At minimum, instrument:

- request and trace IDs;
- ingestion job, source fetch, parse, OCR, chunk, embed, index, and publish
  spans;
- query normalization, BM25, dense retrieval, RRF, reranking, graph expansion,
  context construction, LLM, and citation validation spans;
- document/version IDs and hashes rather than raw sensitive content;
- model/index/config versions;
- latency, token counts, cost estimates, error types, and abstention reasons;
- retrieval and answer evaluation outcomes.

OpenTelemetry-compatible tracing and Prometheus-compatible metrics are the
default implementation choice. A local developer setup may use an
OpenTelemetry collector plus a lightweight trace UI; production deployment
may route the same telemetry to a managed backend. Sensitive payload capture
must be opt-in and redacted.

## Frontend direction

Replace Streamlit with a professional-grade Next.js application. The first
information architecture should support:

- a lawyer workspace with search, filters, current-status badges, and a
  version/change timeline;
- an evidence panel showing source, page/section anchor, dates, and linked
  provisions;
- a comparison view for old versus new text;
- a plain-language explanation mode with a prominent informational-use
  boundary;
- feedback controls that distinguish “wrong source,” “stale status,”
  “missing authority,” and “poor explanation.”

The frontend should consume typed API contracts from the backend rather than
reimplementing retrieval logic.

## Repository skill files

After this design is approved, create a repository-level `skills/` directory
containing:

- `copilot-instructions.md`: concise project-wide instructions, safety
  boundaries, source/citation rules, and development commands;
- `decisions.md`: append-only architecture and product decisions with reasons,
  alternatives considered, and dates;
- `frontend.md`: Next.js conventions, UX, API contracts, accessibility, and
  legal-information boundaries;
- `backend.md`: FastAPI/service conventions, data contracts, error handling,
  security, and testing;
- `rag.md`: corpus model, ingestion, BM25, HNSW, RRF, reranking, evaluation,
  observability, and provenance requirements.

These files are operating instructions for future agents, not a substitute
for tests or runtime validation.

## Goals

- Establish an authoritative, version-aware criminal-law corpus.
- Make source provenance and effective dates visible in every answer.
- Add reproducible RAG evaluation before tuning production retrieval.
- Add traceable observability across ingestion and query execution.
- Move the user experience to Next.js without discarding the working API
  boundary.
- Support safe abstention and clear public-facing explanations.

## Non-goals

- Providing personalized legal advice or predicting case outcomes.
- Covering every Indian jurisdiction or every legal domain in the first
  release.
- Treating unofficial aggregators as primary authority.
- Building a legal case-management system.
- Optimizing for benchmark scores without legal-reviewable evidence.

## Anti-goals

The project fails if it:

- cites a stale or superseded provision as current without warning;
- hides uncertainty behind a single confidence number;
- answers from a plausible chunk without showing the authoritative source;
- silently swallows ingestion, retrieval, or citation-validation failures;
- stores sensitive user material by default;
- expands scope to all Indian law before the first corpus and evaluation set
  are reliable.

## Constraints and builder calls

- **Builder call:** retain FastAPI and Python for the initial backend migration
  to minimize risk while introducing retrieval interfaces and better
  instrumentation.
- **Builder call:** use TypeScript and Next.js for the new frontend, with the
  exact UI component library selected during implementation.
- **Builder call:** start with English primary sources; add Hindi or other
  Indian-language support only after provenance and evaluation workflows are
  stable.
- **Builder call:** keep Pinecone behind an adapter while evaluating BM25,
  HNSW, and hybrid retrieval; do not force an immediate data migration.
- **Constraint:** official source availability, terms, rate limits, and
  redistribution rights must be verified before automated production
  ingestion.
- **Constraint:** legal-reviewable citations and temporal correctness take
  priority over answer fluency.

## Resolved design decisions

The following defaults resolve the builder-check questions for the first
release:

- **Source conflicts:** the Gazette or other primary legal instrument
  controls legal effect; India Code controls the consolidated-text
  representation; Ministry of Home Affairs and PIB material is explanatory.
  A contradiction is retained as a visible conflict and causes abstention
  for claims that depend on the conflict.
- **Corpus boundary:** cover the central BNS, BNSS, and BSA corpus; retain the
  corresponding IPC, CrPC, and Indian Evidence Act material needed for the
  crosswalk; ingest directly linked central commencement and amendment
  instruments from 2023 onward; exclude state law from the first release.
- **Commencement model:** represent provision-level effective intervals,
  exceptions, territories, conditions, and unknown status explicitly rather
  than reducing them to a single current boolean.
- **Answer gate:** answer only when required source anchors and temporal
  constraints are satisfied. Otherwise expose the missing evidence and
  abstain or ask for clarification.
- **Pilot scale:** optimize for a small pilot used by one legal team, with
  asynchronous ingestion and an interactive answer target below eight
  seconds at p95. Correctness and traceability take precedence over
  throughput.

## Open implementation questions

The builder should resolve these after approval:

- exact India Code and e-Gazette acquisition method and refresh schedule;
- storage choice for original files, normalized documents, metadata, and
  evaluation artifacts;
- HNSW engine and deployment topology;
- trace/metrics backend for local and production environments;
- authentication, tenancy, retention, and document-upload policy;
- the first curated Supreme Court judgment set and legal-review process.
