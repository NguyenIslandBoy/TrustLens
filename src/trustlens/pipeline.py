"""
src/trustlens/pipeline.py
===========================
End-to-end pipeline orchestrator for TrustLens.

Stages:
  1. Fetch paper metadata from OpenAlex
  2. Download PDFs
  3. Extract text from PDFs
  4. Segment methods sections
  5. Extract methodology via local LLM
  6. Store in MongoDB
  7. Export to DuckDB

Run:
    python -m src.trustlens.pipeline
    python -m src.trustlens.pipeline --max-papers 10 --skip-download
"""

import argparse
import logging
import sys
from pathlib import Path

from openai import OpenAI

import config
from src.trustlens.ingest.openalex import fetch_papers
from src.trustlens.ingest.pdf_fetcher import batch_download
from src.trustlens.parse.extractor import extract_text
from src.trustlens.parse.segmenter import segment
from src.trustlens.llm.batch_runner import extract_methodology
from src.trustlens.store.database import (
    get_mongo_collection,
    build_paper_document,
    upsert_paper,
    ensure_indexes,
    export_to_duckdb,
)

log = logging.getLogger("trustlens.pipeline")


def run_pipeline(
    max_papers: int = 50,
    skip_download: bool = False,
    skip_extraction: bool = False,
    export: bool = True,
) -> dict:
    """
    Run the full TrustLens pipeline.

    Args:
        max_papers:       Max papers to fetch from OpenAlex
        skip_download:    Skip PDF download (use already-downloaded files)
        skip_extraction:  Skip LLM extraction (useful for testing ingestion only)
        export:           Export to DuckDB at the end

    Returns:
        Summary dict with counts per stage
    """
    logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(config.BASE_DIR / "logs" / "pipeline.log"),
    ],
    )
    summary = {
        "papers_fetched":    0,
        "pdfs_downloaded":   0,
        "pdfs_failed":       0,
        "texts_extracted":   0,
        "llm_succeeded":     0,
        "llm_failed":        0,
        "mongo_upserted":    0,
        "duckdb_exported":   0,
    }

    # -----------------------------------------------------------------------
    # Stage 1 — Fetch metadata
    # -----------------------------------------------------------------------
    log.info("=" * 60)
    log.info("Stage 1 — Fetching paper metadata from OpenAlex")
    log.info("=" * 60)

    papers = fetch_papers(
        email=config.OPENALEX_EMAIL,
        max_results=max_papers,
    )
    summary["papers_fetched"] = len(papers)
    log.info(f"Fetched {len(papers)} papers")

    if not papers:
        log.error("No papers fetched — aborting pipeline")
        return summary

    # -----------------------------------------------------------------------
    # Stage 2 — Download PDFs
    # -----------------------------------------------------------------------
    if not skip_download:
        log.info("=" * 60)
        log.info("Stage 2 — Downloading PDFs")
        log.info("=" * 60)

        successes, failures = batch_download(papers, config.PDF_DIR)
        summary["pdfs_downloaded"] = len(successes)
        summary["pdfs_failed"] = len(failures)
        log.info(f"Downloaded {len(successes)}, failed {len(failures)}")
    else:
        log.info("Stage 2 — Skipped (--skip-download)")

    # -----------------------------------------------------------------------
    # Stage 3+4 — Extract text and segment
    # -----------------------------------------------------------------------
    log.info("=" * 60)
    log.info("Stage 3+4 — Extracting text and segmenting methods sections")
    log.info("=" * 60)

    segments = []
    for paper in papers:
        pdf_filename = paper.openalex_id.rstrip("/").split("/")[-1] + ".pdf"
        pdf_path = config.PDF_DIR / pdf_filename

        extracted = extract_text(pdf_path, paper.openalex_id)
        if extracted.success:
            summary["texts_extracted"] += 1

        seg = segment(extracted, abstract=paper.abstract)
        seg._paper_meta = paper   # attach for downstream use
        segments.append(seg)

    log.info(f"Texts extracted: {summary['texts_extracted']}/{len(papers)}")

    if skip_extraction:
        log.info("Stage 5 — Skipped (--skip-extraction)")
        return summary

    # -----------------------------------------------------------------------
    # Stage 5 — LLM extraction
    # -----------------------------------------------------------------------
    log.info("=" * 60)
    log.info("Stage 5 — LLM methodology extraction")
    log.info("=" * 60)

    client = OpenAI(base_url=config.OLLAMA_BASE_URL, api_key="ollama")
    collection = get_mongo_collection(config.MONGO_URI, config.MONGO_DB)
    ensure_indexes(collection)

    for i, seg in enumerate(segments, 1):
        paper = seg._paper_meta
        log.info(f"[{i}/{len(segments)}] {paper.title[:55]}...")

        result = extract_methodology(seg, client, config.OLLAMA_MODEL)

        if result.success:
            summary["llm_succeeded"] += 1
        else:
            summary["llm_failed"] += 1
            log.warning(f"Extraction failed: {result.error}")

        # Stage 6 — Store in MongoDB regardless of LLM success
        doc = build_paper_document(paper, result, seg)
        if upsert_paper(collection, doc):
            summary["mongo_upserted"] += 1

    log.info(f"LLM succeeded: {summary['llm_succeeded']}, failed: {summary['llm_failed']}")
    log.info(f"MongoDB upserted: {summary['mongo_upserted']}")

    # -----------------------------------------------------------------------
    # Stage 7 — Export to DuckDB
    # -----------------------------------------------------------------------
    if export:
        log.info("=" * 60)
        log.info("Stage 7 — Exporting to DuckDB")
        log.info("=" * 60)

        count = export_to_duckdb(collection, config.DUCKDB_PATH)
        summary["duckdb_exported"] = count
        log.info(f"Exported {count} rows to {config.DUCKDB_PATH}")

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------
    log.info("=" * 60)
    log.info("Pipeline complete")
    for k, v in summary.items():
        log.info(f"  {k:<25} {v}")
    log.info("=" * 60)

    return summary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="TrustLens extraction pipeline")
    parser.add_argument("--max-papers",      type=int, default=config.MAX_PAPERS)
    parser.add_argument("--skip-download",   action="store_true")
    parser.add_argument("--skip-extraction", action="store_true")
    parser.add_argument("--no-export",       action="store_true")
    args = parser.parse_args()

    run_pipeline(
        max_papers=args.max_papers,
        skip_download=args.skip_download,
        skip_extraction=args.skip_extraction,
        export=not args.no_export,
    )


if __name__ == "__main__":
    main()