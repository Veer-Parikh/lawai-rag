from llm.gemini_client import _validate_citations


def test_structured_citation_must_match_retrieved_evidence():
    valid, warnings = _validate_citations(
        ["Section 999, Imaginary Act"],
        [{"act_name": "Bharatiya Nyaya Sanhita", "section_number": "103"}],
    )

    assert valid == []
    assert warnings == ["Removed unverifiable citation: Section 999, Imaginary Act"]
