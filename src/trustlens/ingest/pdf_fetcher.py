"""
src/trustlens/ingest/pdf_fetcher.py
=====================================
Download open-access PDFs from URLs provided by OpenAlex.

Design decisions:
- Skips already-downloaded PDFs (idempotent — safe to re-run)
- Validates downloaded file is actually a PDF (checks magic bytes)
- Returns explicit success/failure per paper — never raises
"""

import logging
import hashlib
from dataclasses import dataclass
from pathlib import Path

import requests

log = logging.getLogger(__name__)

PDF_MAGIC = b"%PDF"


@dataclass
class DownloadResult:
    openalex_id: str
    success: bool
    path: Path | None = None
    error: str | None = None
    skipped: bool = False   # True if file already existed


def download_pdf(
    openalex_id: str,
    pdf_url: str,
    output_dir: Path,
    timeout: int = 30,
) -> DownloadResult:
    """
    Download a single PDF.

    Args:
        openalex_id: Used to name the file consistently
        pdf_url:     Direct URL to the PDF
        output_dir:  Directory to save into
        timeout:     Request timeout in seconds

    Returns:
        DownloadResult — never raises
    """
    filename = _safe_filename(openalex_id) + ".pdf"
    dest = output_dir / filename

    # Idempotent — skip if already downloaded
    if dest.exists() and dest.stat().st_size > 0:
        log.debug(f"Already exists, skipping: {filename}")
        return DownloadResult(openalex_id=openalex_id, success=True, path=dest, skipped=True)

    try:
        resp = requests.get(
            pdf_url,
            timeout=timeout,
            headers={"User-Agent": "TrustLens/0.1 (academic research pipeline)"},
            allow_redirects=True,
        )
        resp.raise_for_status()
    except requests.RequestException as e:
        log.warning(f"Download failed [{openalex_id}]: {e}")
        return DownloadResult(openalex_id=openalex_id, success=False, error=str(e))

    # Validate it's actually a PDF
    if not resp.content[:4] == PDF_MAGIC:
        log.warning(f"Not a valid PDF [{openalex_id}]: {pdf_url}")
        return DownloadResult(
            openalex_id=openalex_id,
            success=False,
            error="Response is not a valid PDF file",
        )

    dest.write_bytes(resp.content)
    log.info(f"Downloaded: {filename} ({len(resp.content) / 1024:.1f} KB)")
    return DownloadResult(openalex_id=openalex_id, success=True, path=dest)


def batch_download(
    papers: list,        # list[PaperMeta]
    output_dir: Path,
    timeout: int = 30,
) -> tuple[list[DownloadResult], list[DownloadResult]]:
    """
    Download PDFs for a list of PaperMeta objects.

    Returns:
        (successes, failures) — two lists of DownloadResult
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    successes, failures = [], []

    for i, paper in enumerate(papers, 1):
        log.info(f"[{i}/{len(papers)}] {paper.title[:60]}...")
        result = download_pdf(
            openalex_id=paper.openalex_id,
            pdf_url=paper.pdf_url,
            output_dir=output_dir,
            timeout=timeout,
        )
        if result.success:
            successes.append(result)
        else:
            failures.append(result)

    log.info(f"Batch download complete — {len(successes)} succeeded, {len(failures)} failed")
    return successes, failures


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _safe_filename(openalex_id: str) -> str:
    """
    Convert an OpenAlex ID like 'https://openalex.org/W2741809807'
    to a safe filename like 'W2741809807'.
    """
    name = openalex_id.rstrip("/").split("/")[-1]
    # Fallback: hash if ID is somehow weird
    if not name or len(name) > 50:
        name = hashlib.md5(openalex_id.encode()).hexdigest()[:12]
    return name