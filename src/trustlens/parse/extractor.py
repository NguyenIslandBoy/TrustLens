"""
src/trustlens/parse/extractor.py
==================================
Extract raw text from a PDF file using pdfplumber.

Returns text per page so the segmenter can work page-aware.
"""

import logging
from dataclasses import dataclass
from pathlib import Path

import pdfplumber

log = logging.getLogger(__name__)


@dataclass
class ExtractedText:
    openalex_id: str
    pages: list[str]        # raw text per page
    total_pages: int
    success: bool
    error: str | None = None

    @property
    def full_text(self) -> str:
        return "\n\n".join(self.pages)

    @property
    def word_count(self) -> int:
        return len(self.full_text.split())


def extract_text(pdf_path: Path, openalex_id: str) -> ExtractedText:
    """
    Extract text from a PDF file, one entry per page.

    Returns ExtractedText — never raises.
    """
    if not pdf_path.exists():
        return ExtractedText(
            openalex_id=openalex_id,
            pages=[],
            total_pages=0,
            success=False,
            error=f"File not found: {pdf_path}",
        )

    try:
        pages = []
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                text = page.extract_text() or ""
                pages.append(text)

        if not any(p.strip() for p in pages):
            return ExtractedText(
                openalex_id=openalex_id,
                pages=pages,
                total_pages=len(pages),
                success=False,
                error="PDF appears to be scanned/image-based — no extractable text",
            )

        log.info(f"Extracted {len(pages)} pages, {sum(len(p.split()) for p in pages)} words [{openalex_id}]")
        return ExtractedText(
            openalex_id=openalex_id,
            pages=pages,
            total_pages=len(pages),
            success=True,
        )

    except Exception as e:
        log.warning(f"Extraction failed [{openalex_id}]: {e}")
        return ExtractedText(
            openalex_id=openalex_id,
            pages=[],
            total_pages=0,
            success=False,
            error=str(e),
        )