from llm.citation_audit import audit_answer_citations


def test_audit_maps_inline_citation_to_authoritative_source():
    answer = "The offence is defined in Section 103, Bharatiya Nyaya Sanhita."
    chunks = [
        {
            "chunk_id": "bns-103",
            "act_name": "Bharatiya Nyaya Sanhita",
            "section_number": "103",
            "source_url": "https://egazette.nic.in/bns.pdf",
            "page_anchor": "page-12",
        }
    ]

    result = audit_answer_citations(answer, chunks)

    assert result.references[0].verified is True
    assert result.references[0].source_chunk_ids == ["bns-103"]
    assert result.warnings == []


def test_audit_warns_for_unverified_inline_case_citation():
    result = audit_answer_citations(
        "The Supreme Court held this in AIR 2020 SC 999.",
        [{"chunk_id": "judgment-1", "citation": "AIR 2019 SC 100"}],
    )

    assert result.references[0].verified is False
    assert "AIR 2020 SC 999" in result.warnings[0]
