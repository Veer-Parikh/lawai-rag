"""
utils/text_cleaner.py — Normalize raw text extracted from PDFs.
Handles OCR noise, ligatures, inconsistent whitespace, and legal formatting quirks.
"""
import re


# Common ligature replacements from OCR
_LIGATURES = {
    "\ufb01": "fi",
    "\ufb02": "fl",
    "\ufb00": "ff",
    "\ufb03": "ffi",
    "\ufb04": "ffl",
    "\u2019": "'",
    "\u2018": "'",
    "\u201c": '"',
    "\u201d": '"',
    "\u2013": "-",
    "\u2014": "-",
    "\u00a0": " ",  # non-breaking space
}


def clean_text(text: str) -> str:
    """
    Full cleaning pipeline for raw PDF-extracted text.
    Returns a normalized string ready for chunking.
    """
    if not text:
        return ""

    # 1. Replace ligatures and smart quotes
    for char, replacement in _LIGATURES.items():
        text = text.replace(char, replacement)

    # 2. Remove null bytes and control characters (keep newlines and tabs)
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)

    # 3. Normalize unicode whitespace
    text = re.sub(r"\r\n", "\n", text)     # Windows line endings
    text = re.sub(r"\r", "\n", text)       # Old Mac line endings

    # 4. Collapse excessive blank lines (>2 consecutive) to double newline
    text = re.sub(r"\n{3,}", "\n\n", text)

    # 5. Strip trailing whitespace from each line
    text = "\n".join(line.rstrip() for line in text.split("\n"))

    # 6. Normalize runs of spaces (not newlines) to single space
    text = re.sub(r"[ \t]{2,}", " ", text)

    # 7. Remove page-header/footer noise: patterns like "Page 1 of 23", "www.indiacode.nic.in"
    text = re.sub(r"(?im)^page\s+\d+\s+of\s+\d+\s*$", "", text)
    text = re.sub(r"https?://\S+", "", text)

    # 8. Final strip
    return text.strip()


def normalize_section_header(header: str) -> str:
    """
    Normalize a section header string for consistent display.
    e.g. "SECTION 302" → "Section 302"
    """
    return re.sub(r"\s+", " ", header).strip().title()
