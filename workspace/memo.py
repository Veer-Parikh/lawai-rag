"""Render a matter as a research memo in Markdown."""
from __future__ import annotations

from datetime import date
from typing import Any

from workspace.matters import Entry, Matter

STATUS_TEXT = {
    "in_force": "In force",
    "notified": "Notified",
    "enacted": "Enacted",
    "superseded": "Superseded",
    "repealed": "Repealed",
}


def evidence_state(answer: dict[str, Any]) -> str:
    """Same rule the research workspace shows on screen."""
    if answer.get("confidence_tier") == "LOW" or not answer.get("sources"):
        return "Not answered: the corpus did not contain enough evidence"
    unverified = [c for c in answer.get("citations", []) if not c.get("verified")]
    if answer.get("confidence_tier") == "MEDIUM" or unverified or answer.get("citation_warnings"):
        return "Partly supported: some statements could not be matched to a source"
    return "Supported by sources"


def _cell(value: Any) -> str:
    return str(value or "").replace("|", "\\|").replace("\n", " ")


def _entry_markdown(number: int, entry: Entry) -> list[str]:
    answer = entry.answer
    lines = [f"## {number}. {entry.question}", ""]
    meta = [f"Evidence: {evidence_state(answer)}"]
    if entry.as_of:
        meta.append(f"Law as on {entry.as_of}")
    lines += [f"*{'. '.join(meta)}.*", "", answer.get("answer", "").strip(), ""]

    citations = answer.get("citations") or []
    if citations:
        lines += ["**Citations**", ""]
        for citation in citations:
            mark = "matched to a retrieved source" if citation.get("verified") else "NOT found in the retrieved sources"
            urls = ", ".join(citation.get("source_urls") or [])
            lines.append(f"- {citation.get('text')}: {mark}" + (f" ({urls})" if urls else ""))
        lines.append("")

    sources = answer.get("sources") or []
    if sources:
        lines += ["**Sources consulted**", "", "| # | Authority | Section | Status | Effective | Link |", "|---|---|---|---|---|---|"]
        for index, source in enumerate(sources, start=1):
            lines.append(
                f"| {index} | {_cell(source.get('act_name') or source.get('case_name') or source.get('source_file'))} "
                f"| {_cell(source.get('section'))} | {_cell(STATUS_TEXT.get(source.get('legal_status') or '', 'Unknown'))} "
                f"| {_cell(source.get('effective_date'))} | {_cell(source.get('source_url'))} |"
            )
        lines.append("")

    warnings = answer.get("citation_warnings") or []
    if warnings:
        lines += ["**Check before relying on this**", ""] + [f"- {warning}" for warning in warnings] + [""]
    if entry.note.strip():
        lines += ["**Researcher's note**", "", entry.note.strip(), ""]
    return lines


def render_markdown(matter: Matter, *, prepared_by: str, today: date | None = None) -> str:
    today = today or date.today()
    lines = [f"# Research memo: {matter.title}", ""]
    if matter.reference:
        lines.append(f"Reference: {matter.reference}  ")
    lines += [f"Prepared by {prepared_by} on {today:%d %B %Y} with LawAI.", ""]
    lines += [
        "> Research support only. This memo is not legal advice. Verify every authority "
        "against the official text before relying on it.",
        "",
    ]
    if matter.notes.strip():
        lines += ["## Matter notes", "", matter.notes.strip(), ""]
    if not matter.entries:
        lines += ["No research has been saved to this matter yet.", ""]
    for number, entry in enumerate(matter.entries, start=1):
        lines += _entry_markdown(number, entry)
    return "\n".join(lines).rstrip() + "\n"
