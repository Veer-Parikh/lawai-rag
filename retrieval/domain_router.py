"""
retrieval/domain_router.py — Two-stage domain classification for user queries.

Stage 1: Keyword matching (< 1ms) — deterministic, no LLM.
Stage 2: Gemini Flash LLM call (≈200ms) — only for ambiguous queries.
         Results cached via LRU to avoid redundant API calls.
"""
from __future__ import annotations

import hashlib
import json
from functools import lru_cache

import config
from utils.logger import get_logger

logger = get_logger(__name__)

# ─── Keyword Map ──────────────────────────────────────────────────────────────

_DOMAIN_KEYWORDS: dict[str, list[str]] = {
    "criminal": [
        "murder", "fir", "bail", "arrest", "ipc", "bns", "bnss", "crpc", "crime",
        "accused", "offence", "offense", "culpable homicide", "theft", "rape",
        "robbery", "kidnapping", "dowry", "anticipatory bail", "chargesheet",
        "cognizable", "non-cognizable", "section 302", "section 420", "section 498",
        "punishable", "imprisonment", "death penalty", "indian penal code",
        "bharatiya nyaya sanhita", "criminal", "police",
    ],
    "civil": [
        "contract", "breach", "specific performance", "property", "lease",
        "landlord", "tenant", "mortgage", "injunction", "tort", "negligence",
        "damages", "civil suit", "plaintiff", "defendant", "decree", "cpc",
        "civil procedure", "partition", "easement", "specific relief",
    ],
    "economic": [
        "consumer", "complaint", "deficiency of service", "e-commerce", "it act",
        "data protection", "cyber", "digital", "information technology",
        "gdpr", "privacy", "personal data", "trademark", "copyright",
        "intellectual property", "competition", "monopoly", "sebi",
    ],
    "family": [
        "divorce", "maintenance", "alimony", "custody", "marriage", "hindu marriage",
        "succession", "inheritance", "will", "testament", "adoption", "guardian",
        "dowry", "domestic violence", "matrimonial", "hindu succession",
        "special marriage", "muslim personal law", "christian divorce",
    ],
    "constitutional": [
        "fundamental rights", "article", "constitutional", "writ", "habeas corpus",
        "mandamus", "certiorari", "quo warranto", "directive principles",
        "preamble", "amendment", "parliament", "legislature", "federalism",
        "supreme court jurisdiction", "high court", "article 21", "article 19",
        "article 32", "article 226", "right to equality",
    ],
    "misc": [
        "labour", "labor", "employee", "employer", "salary", "wages", "pf",
        "provident fund", "esi", "termination", "retrenchment", "gratuity",
        "minimum wages", "industrial dispute", "trade union", "factories act",
        "workmen", "compensation", "state act",
    ],
}


def _keyword_classify(query: str) -> list[str]:
    """
    Match query against keyword lists.
    Returns list of matched domains (may be 0, 1, or several).
    """
    low = query.lower()
    matched: list[str] = []
    for domain, keywords in _DOMAIN_KEYWORDS.items():
        if any(kw in low for kw in keywords):
            matched.append(domain)
    return matched


# ─── LLM Fallback (Stage 2) ───────────────────────────────────────────────────

def _llm_classify(query: str) -> list[str]:
    """
    Call Gemini Flash to classify the query domain.
    Returns list of domain strings.
    """
    import google.generativeai as genai

    genai.configure(api_key=config.GEMINI_API_KEY)
    model = genai.GenerativeModel(config.GEMINI_MODEL_NAME)

    prompt = (
        "Classify this Indian legal query into one or more of the following categories: "
        "criminal, civil, economic, family, constitutional, misc.\n\n"
        f"Query: {query}\n\n"
        'Return JSON only, with no extra text. Format: {"domains": ["..."]}.'
    )

    try:
        response = model.generate_content(
            prompt,
            generation_config={"temperature": 0.0, "max_output_tokens": 64},
        )
        text = response.text.strip()
        # Strip markdown code fences if present
        if text.startswith("```"):
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
        parsed = json.loads(text)
        domains = parsed.get("domains", [])
        valid = [d for d in domains if d in _DOMAIN_KEYWORDS or d == "general"]
        return valid or ["general"]
    except Exception as exc:
        logger.warning(f"LLM domain classification failed: {exc} — defaulting to general")
        return ["general"]


@lru_cache(maxsize=config.DOMAIN_LRU_CACHE_SIZE)
def _cached_llm_classify(query_hash: str, query: str) -> tuple[str, ...]:
    """LRU-cached wrapper around _llm_classify. Returns tuple (hashable)."""
    return tuple(_llm_classify(query))


# ─── Public Interface ─────────────────────────────────────────────────────────

class DomainResult:
    __slots__ = ("domains", "namespaces", "stage")

    def __init__(self, domains: list[str], stage: str):
        self.domains = domains
        self.stage = stage  # "keyword" | "llm" | "general"
        self.namespaces = self._build_namespaces()

    def _build_namespaces(self) -> list[str]:
        ns_set: list[str] = []
        seen: set[str] = set()
        for domain in self.domains:
            for ns in config.DOMAIN_TO_NAMESPACES.get(domain, config.ALL_NAMESPACES):
                if ns not in seen:
                    ns_set.append(ns)
                    seen.add(ns)
        return ns_set if ns_set else config.ALL_NAMESPACES


def classify_query(query: str) -> DomainResult:
    """
    Two-stage domain classification.

    Stage 1: keyword matching.
    Stage 2: LLM fallback if 0 or ≥3 domains matched.

    Returns:
        DomainResult with domains, namespaces, and stage used.
    """
    stage1_domains = _keyword_classify(query)

    # Exactly 1 or 2 domain matches → confident, use directly
    if 1 <= len(stage1_domains) <= 2:
        logger.info(f"Domain router: keyword stage matched {stage1_domains}")
        return DomainResult(domains=stage1_domains, stage="keyword")

    # 0 matches or ≥ 3 matches → ambiguous → LLM
    query_hash = hashlib.sha256(query.encode()).hexdigest()
    llm_domains = list(_cached_llm_classify(query_hash, query))
    logger.info(f"Domain router: LLM stage classified {llm_domains}")

    if not llm_domains or llm_domains == ["general"]:
        return DomainResult(domains=["general"], stage="general")

    return DomainResult(domains=llm_domains, stage="llm")
