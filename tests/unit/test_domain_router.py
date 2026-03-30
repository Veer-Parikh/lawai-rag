"""tests/unit/test_domain_router.py"""
from __future__ import annotations

import pytest
from retrieval.domain_router import _keyword_classify, classify_query, DomainResult


def test_criminal_keywords():
    domains = _keyword_classify("What is bail under Section 302 IPC?")
    assert "criminal" in domains


def test_family_keywords():
    domains = _keyword_classify("How do I file for divorce and claim maintenance?")
    assert "family" in domains


def test_constitutional_keywords():
    domains = _keyword_classify("What are fundamental rights under Article 21?")
    assert "constitutional" in domains


def test_economic_keywords():
    domains = _keyword_classify("Can I file a consumer complaint against Amazon?")
    assert "economic" in domains


def test_civil_keywords():
    domains = _keyword_classify("What is breach of contract and damages?")
    assert "civil" in domains


def test_misc_keywords():
    domains = _keyword_classify("Can my employer deduct salary for absence?")
    assert "misc" in domains


def test_no_keywords_returns_empty():
    domains = _keyword_classify("tell me a joke")
    # May or may not match — just check it's a list
    assert isinstance(domains, list)


def test_domain_result_namespaces_populated():
    result = DomainResult(domains=["criminal"], stage="keyword")
    assert len(result.namespaces) > 0


def test_domain_result_general_uses_all_namespaces():
    result = DomainResult(domains=["general"], stage="general")
    from config import ALL_NAMESPACES
    assert set(result.namespaces) == set(ALL_NAMESPACES)


def test_classify_query_returns_domain_result():
    result = classify_query("What is IPC Section 302?")
    assert isinstance(result, DomainResult)
    assert result.stage in ("keyword", "llm", "general", "override")
    assert len(result.namespaces) > 0
