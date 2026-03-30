"""
src/trustlens/parse/segmenter.py
==================================
Isolate the methods section from extracted paper text.

Strategy:
  1. Look for section headers matching known methods patterns
  2. Extract text from that header until the next major section
  3. Fall back to abstract + first 2000 chars if no methods section found

This is intentionally heuristic — academic papers have wildly
inconsistent formatting. The LLM handles ambiguity downstream.
"""

import logging
import re
from dataclasses import dataclass

from src.trustlens.parse.extractor import ExtractedText

log = logging.getLogger(__name__)

# Section headers that signal the start of a methods section
METHODS_PATTERNS = [
    r"^\s*\d*\.?\s*method(ology|ological|s)?\s*$",
    r"^\s*\d*\.?\s*research design\s*$",
    r"^\s*\d*\.?\s*data and method(s)?\s*$",
    r"^\s*\d*\.?\s*empirical strategy\s*$",
    r"^\s*\d*\.?\s*study design\s*$",
    r"^\s*\d*\.?\s*analytical approach\s*$",
    r"^\s*\d*\.?\s*materials and methods\s*$",
]

# Section headers that signal the END of the methods section
END_PATTERNS = [
    r"^\s*\d*\.?\s*result(s)?\s*$",
    r"^\s*\d*\.?\s*finding(s)?\s*$",
    r"^\s*\d*\.?\s*discussion\s*$",
    r"^\s*\d*\.?\s*conclusion(s)?\s*$",
    r"^\s*\d*\.?\s*analysis\s*$",
    r"^\s*\d*\.?\s*empirical result(s)?\s*$",
]

# Max characters to send to LLM — keep within context window
MAX_SEGMENT_CHARS = 4000


@dataclass
class Segment:
    openalex_id: str
    methods_text: str           # primary content for LLM
    abstract: str | None        # always included for context
    extraction_method: str      # "methods_section" | "fallback"
    char_count: int


def segment(extracted: ExtractedText, abstract: str | None = None) -> Segment:
    """
    Extract the methods section from paper text.

    Args:
        extracted:  ExtractedText from the PDF extractor
        abstract:   Abstract from OpenAlex metadata (used as context)

    Returns:
        Segment with methods_text ready for LLM ingestion
    """
    if not extracted.success or not extracted.full_text.strip():
        # Nothing to segment — return abstract only as fallback
        fallback = abstract or ""
        return Segment(
            openalex_id=extracted.openalex_id,
            methods_text=fallback[:MAX_SEGMENT_CHARS],
            abstract=abstract,
            extraction_method="fallback",
            char_count=len(fallback),
        )

    methods_text = _extract_methods_section(extracted.full_text)

    if methods_text:
        log.info(f"Methods section found [{extracted.openalex_id}] — {len(methods_text)} chars")
        truncated = methods_text[:MAX_SEGMENT_CHARS]
        return Segment(
            openalex_id=extracted.openalex_id,
            methods_text=truncated,
            abstract=abstract,
            extraction_method="methods_section",
            char_count=len(truncated),
        )
        
    # Fallback: abstract + beginning of paper
    log.info(f"No methods section found [{extracted.openalex_id}] — using fallback")
    fallback = _build_fallback(extracted.full_text, abstract)
    truncated_fallback = fallback[:MAX_SEGMENT_CHARS]
    return Segment(
        openalex_id=extracted.openalex_id,
        methods_text=truncated_fallback,
        abstract=abstract,
        extraction_method="fallback",
        char_count=len(truncated_fallback),
    )


def _extract_methods_section(text: str) -> str | None:
    """
    Find and extract the methods section by scanning line by line.
    Returns None if no methods header is found.
    """
    lines = text.split("\n")
    methods_start = None
    methods_end = None

    for i, line in enumerate(lines):
        stripped = line.strip().lower()

        if methods_start is None:
            for pattern in METHODS_PATTERNS:
                if re.match(pattern, stripped, re.IGNORECASE):
                    methods_start = i
                    break
        else:
            for pattern in END_PATTERNS:
                if re.match(pattern, stripped, re.IGNORECASE):
                    methods_end = i
                    break
            if methods_end:
                break

    if methods_start is None:
        return None

    end = methods_end or (methods_start + 150)   # ~150 lines as safety cap
    section_lines = lines[methods_start:end]
    return "\n".join(section_lines).strip()


def _build_fallback(full_text: str, abstract: str | None) -> str:
    """Build a fallback segment from abstract + paper opening."""
    parts = []
    if abstract:
        parts.append(f"Abstract:\n{abstract}")
    parts.append(full_text[:2000])
    combined = "\n\n".join(parts)
    return combined[:MAX_SEGMENT_CHARS]