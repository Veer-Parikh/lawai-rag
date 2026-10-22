# Frontend Skill — Next.js Legal Research Workspace

## Direction

The frontend target is a professional TypeScript Next.js application. The
current Streamlit app is a reference for existing behavior, not the place to
grow the product. Keep the FastAPI API as the initial backend boundary and
consume typed contracts from it.

## First user workflows

### Lawyer workspace

- Search by act, section, phrase, date, document type, and source status.
- Show current-status badges and a provision change timeline.
- Compare prior and current text with amendment/commencement evidence.
- Display citations with publisher, URL, page/section anchor, publication
  date, effective date, and retrieval timestamp.
- Distinguish primary authority from explanatory context.

### Public explanation mode

- Explain a retrieved provision in plain language.
- Keep citations and dates visible rather than hiding them behind an answer.
- Show a clear informational-use disclaimer and avoid personalized advice.
- Provide a safe response when the question asks for a case-specific legal
  conclusion.

## UI invariants

- Never show an answer without its evidence state: supported, conflicting,
  incomplete, or abstained.
- Never represent vector or model confidence as legal certainty.
- Make stale, superseded, unknown, and in-force statuses visually distinct.
- Preserve keyboard navigation, readable contrast, responsive layouts, and
  screen-reader labels.
- Make “wrong source,” “stale status,” “missing authority,” and “poor
  explanation” feedback available.

## Integration conventions

- Generate or maintain typed request/response models from the backend contract.
- Keep retrieval, temporal filtering, and citation validation in the backend.
- Handle loading, partial results, abstention, conflict, and service-error
  states explicitly.
- Do not expose API keys or source-ingestion credentials in browser code.
- Do not add a second legal interpretation pipeline in the frontend.

## Migration rule

Implement new screens in Next.js first. Preserve the Streamlit app until the
Next.js flows cover the approved pilot workflows and have passed accessibility,
API-contract, and citation-display checks; then remove Streamlit deliberately.
