"""Smoke test — hits the real OpenAlex API, so requires internet."""
import pytest
from src.trustlens.ingest.openalex import fetch_papers, _reconstruct_abstract
from unittest.mock import patch, MagicMock
from pathlib import Path
import tempfile
from src.trustlens.ingest.pdf_fetcher import download_pdf, _safe_filename, DownloadResult


def test_fetch_returns_papers():
    papers = fetch_papers(email="test@example.com", max_results=5)
    assert len(papers) > 0


def test_papers_have_pdf_urls():
    papers = fetch_papers(email="test@example.com", max_results=5)
    assert all(p.pdf_url is not None for p in papers)


def test_papers_have_titles():
    papers = fetch_papers(email="test@example.com", max_results=5)
    assert all(len(p.title) > 0 for p in papers)


def test_reconstruct_abstract_basic():
    inverted = {"Hello": [0], "world": [1]}
    result = _reconstruct_abstract(inverted)
    assert result == "Hello world"


def test_reconstruct_abstract_none():
    assert _reconstruct_abstract(None) is None
    
class TestSafeFilename:
    def test_extracts_id_from_url(self):
        result = _safe_filename("https://openalex.org/W2741809807")
        assert result == "W2741809807"

    def test_handles_trailing_slash(self):
        result = _safe_filename("https://openalex.org/W2741809807/")
        assert result == "W2741809807"


class TestDownloadPdf:
    def test_skips_existing_file(self, tmp_path):
        # Create a fake existing PDF
        existing = tmp_path / "W123.pdf"
        existing.write_bytes(b"%PDF-fake content")

        with patch("src.trustlens.ingest.pdf_fetcher._safe_filename", return_value="W123"):
            result = download_pdf("https://openalex.org/W123", "http://fake.url/paper.pdf", tmp_path)

        assert result.skipped is True
        assert result.success is True

    def test_rejects_non_pdf_response(self, tmp_path):
        mock_resp = MagicMock()
        mock_resp.content = b"<html>Not a PDF</html>"
        mock_resp.raise_for_status = MagicMock()

        with patch("requests.get", return_value=mock_resp):
            result = download_pdf("https://openalex.org/W999", "http://fake.url/notpdf", tmp_path)

        assert result.success is False
        assert "not a valid pdf" in result.error.lower()

    def test_handles_network_error_gracefully(self, tmp_path):
        import requests as req
        with patch("requests.get", side_effect=req.RequestException("timeout")):
            result = download_pdf("https://openalex.org/W999", "http://fake.url/paper.pdf", tmp_path)

        assert result.success is False
        assert result.error is not None

    def test_successful_download(self, tmp_path):
        mock_resp = MagicMock()
        mock_resp.content = b"%PDF-1.4 fake pdf content"
        mock_resp.raise_for_status = MagicMock()

        with patch("requests.get", return_value=mock_resp):
            result = download_pdf("https://openalex.org/W2741809807", "http://fake.url/paper.pdf", tmp_path)

        assert result.success is True
        assert result.path.exists()