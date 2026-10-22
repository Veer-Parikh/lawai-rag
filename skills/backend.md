# Backend Skill — FastAPI and Legal Evidence Services

## Service boundary

Keep FastAPI as the initial API boundary while the retrieval and ingestion
internals evolve. Public endpoints should expose typed schemas and explicit
evidence status, not provider-specific Pinecone objects.

## Core contracts

Every answer-capable request and response should be traceable to:

- request and trace IDs;
- normalized query and detected act/section/date constraints;
- source document/version IDs and immutable document hashes;
- section, paragraph, or page anchors;
- publication, notification, and effective dates;
- source status and conflict state;
- retrieval, reranker, prompt, and model versions;
- abstention reason when no safe answer is possible.

## Error and safety behavior

- Validate jurisdiction, dates, section identifiers, and pagination at the
  boundary.
- Return explicit 4xx/5xx errors for invalid input and dependency failures;
  never return a success-shaped empty answer.
- If required evidence is missing, conflicting, or temporally incompatible,
  return an explicit abstention/clarification response.
- Keep legal-information disclaimers in the API response model so clients
  cannot accidentally omit them.
- Redact raw query/document payloads from normal logs.
- Protect admin ingestion and source credentials; do not put secrets in
  frontend bundles or committed files.

## Ingestion and publication

- Preserve original files and normalized text.
- Record publisher, source URL, retrieval time, hash, extraction/OCR status,
  language, document type, and all known legal dates.
- Do not publish an automatically extracted document as authoritative without
  validation status and an audit trail.
- Support asynchronous ingestion jobs with observable states: queued,
  fetching, parsing, chunking, indexing, validating, published, failed.
- Surface partial failures rather than silently dropping a source.

## Observability

Use correlated OpenTelemetry-compatible spans and metrics for request,
ingestion, retrieval, LLM, citation validation, and failure paths. Retain the
existing JSONL logger only as a compatibility or local fallback, not as the
sole production observability layer.

At minimum record latency, tokens, cost estimate, error type, index/config
versions, retrieval stage, citation warnings, and abstention reason without
capturing sensitive content by default.

## Testing

- Unit-test schemas, temporal status logic, source precedence, abstention, and
  citation validation.
- Integration-test chat/search endpoints with provider boundaries isolated.
- Add regression cases for stale provisions, conflicting official sources,
  missing anchors, OCR uncertainty, and invalid dates.
- Run `pytest tests\ -v` before completion.
