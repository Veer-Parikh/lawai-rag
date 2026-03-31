"""
llm/gemini_client.py — Gemini 2.5 Flash API wrapper with citation post-processing.

Sends the assembled prompt to Gemini, parses JSON output,
validates citations against retrieved metadata, and strips unverifiable ones.
"""
from __future__ import annotations

import json
import re
import time
from typing import Any

# Heavy import moved inside functions to prevent blocking startup
# import google.generativeai as genai

import config
from utils.logger import get_logger
from utils.exceptions import GeminiError

logger = get_logger(__name__)

# Indian legal citation regex patterns
_CITATION_REGEX = re.compile(
    r"(?:AIR\s+\d{4}\s+SC\s+\d+|\(\d{4}\)\s+\d+\s+SCC\s+\d+|JT\s+\d{4}\s+\(\d+\)\s+SC\s+\d+)"
)
_SECTION_REGEX = re.compile(
    r"(?:Section|Article|Clause)\s+\d+[\w\.\-]*(?:,\s*[\w\s]+)?",
    re.IGNORECASE,
)

# Global for reuse after first lazy-load
_genai_configured = False


def _parse_gemini_json(raw_text: str) -> dict:
    """Parse Gemini JSON output; strip markdown fences if present."""
    text = raw_text.strip()
    if text.startswith("```"):
        parts = text.split("```")
        for part in parts:
            part = part.strip()
            if part.startswith("json"):
                part = part[4:].strip()
            try:
                return json.loads(part)
            except json.JSONDecodeError:
                continue
    return json.loads(text)


def _validate_citations(
    relevant_sections: list[str],
    chunks: list[dict[str, Any]],
) -> tuple[list[str], list[str]]:
    """
    Validate each item in relevant_sections against retrieved chunk metadata.
    
    Checks:
      A) Verbatim match in chunk's citation, act_name+section_number, or section_heading
      B) Matches a known Indian legal citation/section regex

    Returns:
        (valid_sections, citation_warnings)
    """
    # Build a set of all known source strings from retrieved chunks
    known_sources: set[str] = set()
    for chunk in chunks:
        if chunk.get("citation"):
            known_sources.add(chunk["citation"].lower())
        if chunk.get("act_name") and chunk.get("section_number"):
            known_sources.add(
                f"section {chunk['section_number']}, {chunk['act_name']}".lower()
            )
        if chunk.get("section_heading"):
            known_sources.add(chunk["section_heading"].lower())
        if chunk.get("case_name"):
            known_sources.add(chunk["case_name"].lower())

    valid: list[str] = []
    warnings: list[str] = []

    for ref in relevant_sections:
        ref_low = ref.lower().strip()
        # Check A: verbatim in known sources
        if any(ref_low in src or src in ref_low for src in known_sources):
            valid.append(ref)
            continue
        # Check B: matches known Indian legal format
        if _CITATION_REGEX.search(ref) or _SECTION_REGEX.search(ref):
            valid.append(ref)
            continue
        # Neither check passed — strip it
        warnings.append(f"Removed unverifiable citation: {ref}")
        logger.warning(f"Citation validation failed: '{ref}'")

    return valid, warnings


def generate_answer(
    system_prompt: str,
    user_prompt: str,
    chunks: list[dict[str, Any]],
    max_retries: int = 2,
) -> dict:
    """
    Call Gemini 2.5 Flash and return a validated structured response.

    Args:
        system_prompt: Static system instruction string.
        user_prompt: Dynamic user-turn prompt (context + query + format instruction).
        chunks: All chunks used in context (for citation validation).
        max_retries: Number of retries on parse failure.

    Returns:
        Dict with keys: answer, relevant_sections, legal_explanation,
                        disclaimer, citation_warnings, tokens_used.
    """
    last_exc: Exception | None = None

    global _genai_configured
    import google.generativeai as genai

    if not _genai_configured:
        genai.configure(api_key=config.GEMINI_API_KEY)
        _genai_configured = True

    for attempt in range(max_retries + 1):
        try:
            t0 = time.time()
            model = genai.GenerativeModel(
                model_name=config.GEMINI_MODEL_NAME,
                system_instruction=system_prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=config.GEMINI_TEMPERATURE,
                    top_p=config.GEMINI_TOP_P,
                    max_output_tokens=config.GEMINI_MAX_OUTPUT_TOKENS,
                    response_mime_type="application/json",
                ),
            )
            response = model.generate_content(user_prompt)

            latency_ms = int((time.time() - t0) * 1000)
            raw_text = response.text

            parsed = _parse_gemini_json(raw_text)

            # Extract required fields with safe defaults
            answer = parsed.get("answer", "")
            relevant_sections = parsed.get("relevant_sections", [])
            legal_explanation = parsed.get("legal_explanation", "")
            disclaimer = parsed.get(
                "disclaimer",
                "This response is for informational purposes only and does not "
                "constitute legal advice. Consult a qualified lawyer for your specific situation.",
            )

            # Citation post-processing
            valid_sections, citation_warnings = _validate_citations(
                relevant_sections, chunks
            )

            # Token usage
            tokens_used = 0
            try:
                tokens_used = response.usage_metadata.total_token_count
            except Exception:
                pass

            logger.info(
                f"Gemini response: {latency_ms}ms, tokens={tokens_used}, "
                f"citations_stripped={len(citation_warnings)}"
            )

            return {
                "answer": answer,
                "relevant_sections": valid_sections,
                "legal_explanation": legal_explanation,
                "disclaimer": disclaimer,
                "citation_warnings": citation_warnings,
                "tokens_used": tokens_used,
            }

        except (json.JSONDecodeError, KeyError) as exc:
            logger.warning(f"Gemini JSON parse failed (attempt {attempt + 1}): {exc}")
            last_exc = exc
            if attempt < max_retries:
                time.sleep(1.0)
        except Exception as exc:
            logger.error(f"Gemini API error (attempt {attempt + 1}): {exc}")
            last_exc = exc
            if attempt < max_retries:
                time.sleep(2.0 * (attempt + 1))  # exponential backoff

    raise GeminiError(f"Gemini call failed after {max_retries + 1} attempts: {last_exc}")
