"""
ingestion/pdf_loader.py — Extract raw text from PDFs with OCR fallback.

Returns a list of page dicts: {page_num, raw_text, source_path}
OCR is triggered when pdfplumber returns empty text for a page (image-based PDF).
"""
from __future__ import annotations

import io
from pathlib import Path
from typing import TypedDict

import pdfplumber

from utils.logger import get_logger

logger = get_logger(__name__)


class PageData(TypedDict):
    page_num: int
    raw_text: str
    source_path: str


def _ocr_page(page: "pdfplumber.page.Page") -> str:
    """Render a PDF page to image and run Tesseract OCR."""
    try:
        import pytesseract
        from PIL import Image

        img = page.to_image(resolution=200).original
        # Convert to PIL Image if needed
        if not isinstance(img, Image.Image):
            img = Image.fromarray(img)
        text = pytesseract.image_to_string(img, lang="eng")
        return text
    except ImportError:
        logger.warning("pytesseract or Pillow not installed — OCR unavailable for this page.")
        return ""
    except Exception as exc:
        logger.warning(f"OCR failed on page: {exc}")
        return ""


def load_pdf(pdf_path: str | Path) -> list[PageData]:
    """
    Extract all pages from a PDF.
    Falls back to OCR for pages where pdfplumber returns empty text.

    Args:
        pdf_path: Absolute or relative path to the PDF file.

    Returns:
        List of PageData dicts, one per page.
    """
    pdf_path = Path(pdf_path)
    pages: list[PageData] = []

    try:
        with pdfplumber.open(pdf_path) as pdf:
            for i, page in enumerate(pdf.pages):
                raw_text = page.extract_text() or ""
                raw_text = raw_text.strip()

                if not raw_text:
                    logger.info(f"Empty text on page {i+1} of {pdf_path.name} — attempting OCR.")
                    raw_text = _ocr_page(page)

                pages.append(
                    PageData(
                        page_num=i + 1,
                        raw_text=raw_text,
                        source_path=str(pdf_path),
                    )
                )
    except Exception as exc:
        logger.error(f"Failed to load PDF {pdf_path}: {exc}")
        raise

    logger.info(f"Loaded {len(pages)} pages from {pdf_path.name}")
    return pages


def load_all_pdfs(directory: str | Path) -> dict[str, list[PageData]]:
    """
    Recursively load all PDFs in a directory.

    Returns:
        Dict mapping file path string → list of PageData.
    """
    directory = Path(directory)
    results: dict[str, list[PageData]] = {}
    pdf_files = list(directory.rglob("*.pdf")) + list(directory.rglob("*.PDF"))

    logger.info(f"Found {len(pdf_files)} PDF files under {directory}")
    for pdf_path in pdf_files:
        try:
            results[str(pdf_path)] = load_pdf(pdf_path)
        except Exception as exc:
            logger.error(f"Skipping {pdf_path.name}: {exc}")

    return results
