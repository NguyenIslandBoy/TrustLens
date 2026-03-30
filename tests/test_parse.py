"""Tests for PDF extraction and segmentation."""
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

from src.trustlens.parse.extractor import extract_text, ExtractedText
from src.trustlens.parse.segmenter import segment, _extract_methods_section, Segment


class TestExtractText:
    def test_missing_file_returns_failure(self, tmp_path):
        result = extract_text(tmp_path / "nonexistent.pdf", "W999")
        assert result.success is False
        assert "not found" in result.error.lower()

    def test_never_raises(self, tmp_path):
        # Corrupt file — should not raise
        bad_pdf = tmp_path / "bad.pdf"
        bad_pdf.write_bytes(b"not a real pdf")
        try:
            result = extract_text(bad_pdf, "W999")
            assert result.success is False
        except Exception as e:
            pytest.fail(f"extract_text raised: {e}")

    def test_full_text_joins_pages(self):
        extracted = ExtractedText(
            openalex_id="W1",
            pages=["page one text", "page two text"],
            total_pages=2,
            success=True,
        )
        assert "page one text" in extracted.full_text
        assert "page two text" in extracted.full_text

    def test_word_count(self):
        extracted = ExtractedText(
            openalex_id="W1",
            pages=["hello world", "foo bar baz"],
            total_pages=2,
            success=True,
        )
        assert extracted.word_count == 5


class TestSegmenter:
    def _make_extracted(self, text: str, success: bool = True) -> ExtractedText:
        return ExtractedText(
            openalex_id="W_test",
            pages=[text],
            total_pages=1,
            success=success,
        )

    def test_finds_methods_section(self):
        text = """
Introduction
This paper studies trust.

Methods
We surveyed 1200 participants using the Rosenberg scale.
OLS regression was applied to the data.

Results
We found significant effects.
"""
        result = _extract_methods_section(text)
        assert result is not None
        assert "Rosenberg" in result
        assert "OLS" in result

    def test_stops_at_results_section(self):
        text = """
Methods
We used survey data from 500 participants.

Results
Trust was significantly higher.
"""
        result = _extract_methods_section(text)
        assert result is not None
        assert "Results" not in result

    def test_returns_none_when_no_methods(self):
        text = "Introduction\nThis is a paper.\n\nConclusion\nWe concluded things."
        result = _extract_methods_section(text)
        assert result is None

    def test_fallback_when_no_methods_section(self):
        extracted = self._make_extracted("Some paper text without a methods header.")
        seg = segment(extracted, abstract="Abstract text here.")
        assert seg.extraction_method == "fallback"
        assert isinstance(seg.methods_text, str)

    def test_fallback_on_failed_extraction(self):
        extracted = self._make_extracted("", success=False)
        seg = segment(extracted, abstract="Abstract only.")
        assert seg.extraction_method == "fallback"
        assert "Abstract only" in seg.methods_text

    def test_segment_never_exceeds_max_chars(self):
        long_text = "Methods\n" + ("word " * 10000)
        extracted = self._make_extracted(long_text)
        seg = segment(extracted)
        assert seg.char_count <= 4000