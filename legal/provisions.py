"""Split statute text into numbered sections.

Indian statute PDFs set sections in two layouts:

* India Code: ``58. Facts admitted need not be proved. –– No fact need ...``
* Gazette of India: the marginal heading is interleaved with the body, so a
  section line starts with its first words: ``Punishment 103.(1) Whoever ...``,
  ``Murder. 101. Except in the cases ...``.

India Code also marks inserted sections with an amendment footnote bracket:
``2[113B. Presumption as to dowry death.``.

Both also contain numbered lines that are *not* sections: the arrangement of
sections, amendment footnotes (``1. Subs. by Act 4 of 1938``) and numbered
lists inside schedules. We find every candidate, then keep the longest chain of
candidates whose numbers advance like a statute does, weighting each candidate
by how much text follows it so the real body beats the table of contents.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

# A candidate section start at a line start, optionally preceded by up to five
# words of marginal heading and/or an amendment-footnote bracket ("2[").
_CANDIDATE = re.compile(
    r"(?m)^[ \t]*"
    r"(?P<margin>[A-Z][\w,’'\-]*(?:[ \t]+[\w,’'\-]+){0,4}\.?[ \t]+)?"
    r"(?:\d{1,2}\[)?"
    r"(?P<num>\d{1,3})(?P<suffix>[A-Z]{0,3})\.[ \t]*(?=[A-Z“\"(\[‘'])"
)
_FOOTNOTE_START = re.compile(
    r"^(?:Subs|Ins|Omitted|Rep|Added|Renumbered|Vide|See|Cl|Certain|Words|The\s+words|"
    r"Inserted|Substituted|Now\s+see|For\s+the\s+words)\b"
)
_NOISE_LINE = re.compile(
    r"(?im)^(?:.*THE GAZETTE OF IN.*|Sec\.\s*\d+\]?.*|_{5,}.*|\s*\d{1,3}\s*)$"
)
# India Code: "Facts admitted need not be proved. –– No fact ..."
_INDIA_CODE_HEADING = re.compile(r"^(?P<heading>[A-Z][^\n]{2,160}?)\.\s*(?:––|—|–)\s*")

# Sections may be absent (repealed, renumbered) or inserted with letters.
_MAX_GAP = 4


@dataclass(frozen=True)
class ParsedSection:
    number: str          # "103", "65B"
    heading: str | None  # Marginal heading when the layout makes it recoverable
    text: str            # Section body, whitespace-normalised
    page: int            # 1-based page on which the section starts


def _clean(text: str) -> str:
    text = _NOISE_LINE.sub("", text)
    # Amendment footnotes ("1. Subs. by Act 4 of 1938 ...") sit at page bottoms.
    lines = [
        line for line in text.split("\n")
        if not (re.match(r"^\s*\d{1,2}\.\s*", line) and _FOOTNOTE_START.match(re.sub(r"^\s*\d{1,2}\.\s*", "", line)))
    ]
    return re.sub(r"[ \t]+", " ", "\n".join(lines))


@dataclass(frozen=True)
class _Candidate:
    boundary: int    # Where the previous section ends (start of this line)
    body_start: int  # Where this section's text begins (after "103.")
    num: int
    suffix: str
    margin: str      # Marginal-heading words before the number, if any

    @property
    def key(self) -> tuple[int, str]:
        return (self.num, self.suffix)

    @property
    def label(self) -> str:
        return f"{self.num}{self.suffix}"


def parse_sections(pages: list[str], *, min_sections: int = 5) -> list[ParsedSection]:
    """Return sections in order. Empty when the text does not look like a statute."""
    cleaned = [_clean(page) for page in pages]
    offsets: list[int] = []
    position = 0
    for page in cleaned:
        offsets.append(position)
        position += len(page) + 1
    full = "\n".join(cleaned)

    candidates: list[_Candidate] = []
    for match in _CANDIDATE.finditer(full):
        following = full[match.end(): match.end() + 40]
        if _FOOTNOTE_START.match(following):
            continue
        margin = (match.group("margin") or "").strip()
        if margin and re.search(r"\d", margin):
            continue
        candidates.append(_Candidate(
            boundary=match.start(),
            body_start=match.end(),
            num=int(match.group("num")),
            suffix=match.group("suffix"),
            margin=margin,
        ))
    if not candidates:
        return []

    spans = [
        (candidates[i + 1].boundary if i + 1 < len(candidates) else len(full)) - candidates[i].boundary
        for i in range(len(candidates))
    ]
    weights = [1.0 + min(span, 4000) / 4000 for span in spans]

    # Longest weighted chain where each step advances the section number by a
    # statute-like amount. O(n * window) with a bounded look-back.
    best = list(weights)
    previous = [-1] * len(candidates)
    for i, current in enumerate(candidates):
        for j in range(max(0, i - 400), i):
            earlier = candidates[j]
            advances = current.key > earlier.key and current.num - earlier.num <= _MAX_GAP
            if advances and best[j] + weights[i] > best[i]:
                best[i] = best[j] + weights[i]
                previous[i] = j
    end = max(range(len(candidates)), key=best.__getitem__)
    chain = []
    while end != -1:
        chain.append(end)
        end = previous[end]
    chain.reverse()
    if len(chain) < min_sections:
        return []

    sections: list[ParsedSection] = []
    for position_in_chain, index in enumerate(chain):
        candidate = candidates[index]
        stop = candidates[chain[position_in_chain + 1]].boundary if position_in_chain + 1 < len(chain) else len(full)
        body = re.sub(r"\s+", " ", full[candidate.body_start:stop]).strip()
        heading = candidate.margin[:-1] if candidate.margin.endswith(".") else None
        heading_match = _INDIA_CODE_HEADING.match(body)
        if heading is None and heading_match and len(heading_match.group("heading").split()) <= 20:
            heading = heading_match.group("heading").strip()
            body = body[heading_match.end():].strip()
        page = max(i for i, offset in enumerate(offsets) if offset <= candidate.boundary) + 1
        sections.append(ParsedSection(number=candidate.label, heading=heading, text=body, page=page))
    return sections
