"""Tests for the storage layer — MongoDB and DuckDB."""
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch
from src.trustlens.store.database import (
    build_paper_document,
    upsert_paper,
    _flatten_document,
    _row_to_tuple,
    export_to_duckdb,
)


def _make_paper_meta():
    m = MagicMock()
    m.openalex_id = "https://openalex.org/W123"
    m.title = "Trust and Society"
    m.year = 2021
    m.doi = "10.1234/test"
    m.pdf_url = "https://example.com/paper.pdf"
    m.authors = ["Alice Smith", "Bob Jones"]
    m.concepts = ["Social trust", "Sociology"]
    return m


def _make_extraction_result():
    m = MagicMock()
    m.success = True
    m.data = {
        "study_type": "quantitative",
        "sample_size": 1200,
        "countries": ["Germany", "UK"],
        "trust_measure": "Rosenberg scale",
        "statistical_models": ["OLS regression"],
        "key_variables": ["social trust", "income"],
        "data_sources": ["ESS"],
        "extraction_confidence": "high",
    }
    m.latency_seconds = 3.2
    return m


def _make_segment():
    m = MagicMock()
    m.abstract = "A study of social trust in Europe."
    m.extraction_method = "methods_section"
    return m


class TestBuildPaperDocument:
    def test_contains_required_fields(self):
        doc = build_paper_document(
            _make_paper_meta(),
            _make_extraction_result(),
            _make_segment(),
        )
        for field in ["openalex_id", "title", "year", "methodology",
                      "extraction_success", "extracted_at"]:
            assert field in doc

    def test_methodology_is_nested(self):
        doc = build_paper_document(
            _make_paper_meta(),
            _make_extraction_result(),
            _make_segment(),
        )
        assert isinstance(doc["methodology"], dict)
        assert doc["methodology"]["study_type"] == "quantitative"

    def test_extracted_at_is_iso_string(self):
        doc = build_paper_document(
            _make_paper_meta(),
            _make_extraction_result(),
            _make_segment(),
        )
        assert "T" in doc["extracted_at"]   # ISO format contains T


class TestUpsertPaper:
    def test_returns_true_on_success(self):
        mock_collection = MagicMock()
        mock_collection.update_one.return_value = MagicMock()
        result = upsert_paper(mock_collection, {"openalex_id": "W123"})
        assert result is True

    def test_returns_false_on_pymongo_error(self):
        from pymongo.errors import PyMongoError
        mock_collection = MagicMock()
        mock_collection.update_one.side_effect = PyMongoError("connection refused")
        result = upsert_paper(mock_collection, {"openalex_id": "W123"})
        assert result is False


class TestFlattenDocument:
    def test_flattens_list_fields_to_strings(self):
        doc = build_paper_document(
            _make_paper_meta(),
            _make_extraction_result(),
            _make_segment(),
        )
        flat = _flatten_document(doc)
        assert isinstance(flat["authors"], str)
        assert "Alice Smith" in flat["authors"]
        assert isinstance(flat["statistical_models"], str)
        assert "OLS regression" in flat["statistical_models"]

    def test_handles_empty_lists(self):
        doc = build_paper_document(
            _make_paper_meta(),
            _make_extraction_result(),
            _make_segment(),
        )
        doc["methodology"]["countries"] = []
        flat = _flatten_document(doc)
        assert flat["countries"] == ""

    def test_handles_none_methodology(self):
        doc = build_paper_document(
            _make_paper_meta(),
            _make_extraction_result(),
            _make_segment(),
        )
        doc["methodology"] = None
        flat = _flatten_document(doc)
        assert flat["study_type"] is None


class TestRowToTuple:
    def test_returns_18_element_tuple(self):
        doc = build_paper_document(
            _make_paper_meta(),
            _make_extraction_result(),
            _make_segment(),
        )
        flat = _flatten_document(doc)
        row = _row_to_tuple(flat)
        assert len(row) == 18

    def test_first_element_is_openalex_id(self):
        doc = build_paper_document(
            _make_paper_meta(),
            _make_extraction_result(),
            _make_segment(),
        )
        flat = _flatten_document(doc)
        row = _row_to_tuple(flat)
        assert row[0] == "https://openalex.org/W123"


class TestExportToDuckdb:
    def test_returns_zero_when_no_documents(self, tmp_path):
        mock_collection = MagicMock()
        mock_collection.find.return_value = []
        count = export_to_duckdb(mock_collection, tmp_path / "test.duckdb")
        assert count == 0

    def test_creates_duckdb_file(self, tmp_path):
        mock_collection = MagicMock()
        doc = build_paper_document(
            _make_paper_meta(),
            _make_extraction_result(),
            _make_segment(),
        )
        doc["extraction_success"] = True
        mock_collection.find.return_value = [doc]
        db_path = tmp_path / "test.duckdb"
        export_to_duckdb(mock_collection, db_path)
        assert db_path.exists()

    def test_returns_correct_row_count(self, tmp_path):
        mock_collection = MagicMock()
        doc = build_paper_document(
            _make_paper_meta(),
            _make_extraction_result(),
            _make_segment(),
        )
        doc["extraction_success"] = True
        mock_collection.find.return_value = [doc, doc]
        count = export_to_duckdb(mock_collection, tmp_path / "test.duckdb")
        assert count == 2

class TestPipelineSummaryKeys:
    """Verify the pipeline summary dict has the expected structure."""
    def test_summary_has_required_keys(self):
        expected = [
            "papers_fetched", "pdfs_downloaded", "pdfs_failed",
            "texts_extracted", "llm_succeeded", "llm_failed",
            "mongo_upserted", "duckdb_exported",
        ]
        # Import here to avoid circular issues at collection time
        from src.trustlens.pipeline import run_pipeline
        from unittest.mock import patch

        with patch("src.trustlens.pipeline.fetch_papers", return_value=[]), \
             patch("src.trustlens.pipeline.batch_download", return_value=([], [])):
            summary = run_pipeline(max_papers=0, skip_extraction=True, export=False)

        for key in expected:
            assert key in summary