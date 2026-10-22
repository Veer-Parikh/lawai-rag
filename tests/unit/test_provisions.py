from __future__ import annotations

from ingestion.statute_chunker import chunk_statute
from legal.provisions import parse_sections

INDIA_CODE_PAGES = [
    # Arrangement of sections: numbered lines with no bodies, must be ignored.
    "ARRANGEMENT OF SECTIONS\n1. Short title.\n2. Definitions.\n3. Evidence.\n4. Burden.\n5. Proof.\n6. Facts.",
    "1. Short title. –– This Act may be called the Example Evidence Act. It extends to the whole of India.\n"
    "2. Definitions. –– In this Act, unless there is anything repugnant in the subject or context, "
    "the following words have the meanings given.\n"
    "3. Evidence may be given of facts in issue. –– Evidence may be given in any suit or proceeding.\n"
    "1. Subs. by Act 4 of 1938, s. 2, for certain words.\n",
    "4. Burden of proof. –– Whoever desires any Court to give judgment as to any legal right must prove it.\n"
    "2[4A. Presumption as to dowry death. –– When the question is whether a person has committed it.]\n"
    "5. Proof of facts. –– All facts may be proved by oral evidence.\n"
    "6. Facts which are the occasion of facts in issue. –– Facts which are the occasion are relevant.\n",
]

GAZETTE_PAGES = [
    "THE BHARATIYA EXAMPLE SANHITA, 2023\n"
    "Short title. 1. (1) This Sanhita may be called the Bharatiya Example Sanhita.\n"
    "Definitions 2. In this Sanhita, unless the context otherwise requires,–– (1) act denotes a series.\n"
    "Murder. 3. Except in the cases hereinafter excepted, culpable homicide is murder.\n",
    "2 THE GAZETTE OF INDIA EXTRAORDINARY [Part II—\n"
    "Punishment 4.(1) Whoever commits murder shall be punished with death or imprisonment\n"
    "for murder. for life, and shall also be liable to fine.\n"
    "Causing death 5.(1) Whoever causes death of any person by doing any rash or negligent act.\n"
    "Theft. 6. Whoever, intending to take dishonestly any movable property commits theft.\n",
]


def test_india_code_layout_skips_contents_and_footnotes():
    sections = parse_sections(INDIA_CODE_PAGES)

    assert [s.number for s in sections] == ["1", "2", "3", "4", "4A", "5", "6"]
    assert sections[0].heading == "Short title"
    assert sections[4].heading == "Presumption as to dowry death"
    assert "Subs. by Act" not in sections[2].text
    assert sections[3].page == 3


def test_gazette_layout_with_interleaved_marginal_notes():
    sections = parse_sections(GAZETTE_PAGES)

    assert [s.number for s in sections] == ["1", "2", "3", "4", "5", "6"]
    assert sections[2].heading == "Murder"
    assert sections[3].text.startswith("(1) Whoever commits murder")
    # The marginal-note prefix of the next line must not leak into this section.
    assert "Causing death" not in sections[3].text


def test_text_without_sections_returns_nothing():
    assert parse_sections(["A press release about the new criminal laws.", "No numbered sections."]) == []


def test_chunker_uses_numbered_sections_when_no_section_headers():
    pages = [{"page_num": i + 1, "raw_text": text, "source_path": "bns.pdf"} for i, text in enumerate(GAZETTE_PAGES)]

    chunks = chunk_statute(pages, act_name="Bharatiya Example Sanhita")

    assert [c["section_number"] for c in chunks] == ["1", "2", "3", "4", "5", "6"]
    assert chunks[2]["section_heading"] == "Section 3. Murder"
