from ingestion.metadata_builder import build_statute_metadata


def test_statute_metadata_defaults_to_in_force_for_published_corpus():
    metadata = build_statute_metadata(
        {
            "chunk_id": "bns-1",
            "chunk_index": 0,
            "text": "Section 1",
            "section_number": "1",
        },
        "data/statutes/criminal/BNS.pdf",
        1,
        "Bharatiya Nyaya Sanhita",
    )

    assert metadata["legal_status"] == "in_force"
