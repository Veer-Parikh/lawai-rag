"""
retrieval/query_rewriter.py — Rewrite follow-up questions as standalone queries.

Only invoked when session_id is provided and prior queries exist in the session.
Uses a cheap Gemini Flash call to make follow-up questions self-contained.
"""
from __future__ import annotations

import threading

import config
from utils.logger import get_logger

logger = get_logger(__name__)

# ─── In-memory session store ──────────────────────────────────────────────────
# Dict[session_id → List[str]] (list of queries, oldest first, capped at SESSION_MAX_TURNS)
_session_store: dict[str, list[str]] = {}
_session_lock = threading.Lock()


def get_session_history(session_id: str) -> list[str]:
    with _session_lock:
        return list(_session_store.get(session_id, []))


def add_to_session(session_id: str, query: str) -> None:
    with _session_lock:
        history = _session_store.setdefault(session_id, [])
        history.append(query)
        if len(history) > config.SESSION_MAX_TURNS:
            _session_store[session_id] = history[-config.SESSION_MAX_TURNS :]


def clear_session(session_id: str) -> None:
    with _session_lock:
        _session_store.pop(session_id, None)


def rewrite_query(current_query: str, session_id: str | None) -> tuple[str, bool]:
    """
    Rewrite current_query as a standalone question if a session history exists.

    Returns:
        (rewritten_query, was_rewritten)
    """
    if not session_id:
        return current_query, False

    history = get_session_history(session_id)
    if not history:
        return current_query, False

    previous_query = history[-1]

    # Quick heuristic: if query is long/self-contained, skip LLM call
    words = current_query.split()
    if len(words) >= 10:
        return current_query, False

    rewritten = _llm_rewrite(previous_query, current_query)
    if rewritten and rewritten.lower() != current_query.lower():
        logger.info(f"Query rewritten: '{current_query}' → '{rewritten}'")
        return rewritten, True

    return current_query, False


def _llm_rewrite(previous_query: str, current_query: str) -> str:
    """Call Gemini Flash to produce a standalone version of the follow-up."""
    import google.generativeai as genai

    genai.configure(api_key=config.GEMINI_API_KEY)
    model = genai.GenerativeModel(config.GEMINI_MODEL_NAME)

    prompt = (
        "You are helping rewrite Indian legal follow-up questions.\n"
        f"Previous question: {previous_query}\n"
        f"Follow-up question: {current_query}\n\n"
        "Rewrite the follow-up as a complete, self-contained legal question. "
        "Return ONLY the rewritten question, nothing else."
    )

    try:
        response = model.generate_content(
            prompt,
            generation_config={"temperature": 0.0, "max_output_tokens": 128},
        )
        return response.text.strip()
    except Exception as exc:
        logger.warning(f"Query rewriting failed: {exc} — using original query")
        return current_query
