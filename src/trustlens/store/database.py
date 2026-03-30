"""
src/trustlens/store/database.py
=================================
Two-layer storage:
  - MongoDB: live operational store, one document per paper
  - DuckDB:  versioned export for public release / analysis

MongoDB document schema mirrors the extraction schema plus pipeline metadata.
DuckDB export flattens arrays into comma-separated strings for portability.
"""

import logging
import json
from datetime import datetime, timezone
from pathlib import Path

import duckdb
from pymongo import MongoClient, ASCENDING
from pymongo.collection import Collection
from pymongo.errors import PyMongoError

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# MongoDB
# ---------------------------------------------------------------------------

def get_mongo_collection(uri: str, db_name: str, collection: str = "papers") -> Collection:
    """Return a MongoDB collection handle."""
    client = MongoClient(uri)
    return client[db_name][collection]


def upsert_paper(collection: Collection, paper_doc: dict) -> bool:
    """
    Insert or update a paper document by openalex_id.
    Returns True on success, False on failure.
    """
    try:
        collection.update_one(
            {"openalex_id": paper_doc["openalex_id"]},
            {"$set": paper_doc},
            upsert=True,
        )
        log.debug(f"Upserted: {paper_doc['openalex_id']}")
        return True
    except PyMongoError as e:
        log.error(f"MongoDB upsert failed [{paper_doc.get('openalex_id')}]: {e}")
        return False


def build_paper_document(
    paper_meta,         # PaperMeta
    extraction_result,  # ExtractionResult
    segment,            # Segment
) -> dict:
    """
    Assemble the full MongoDB document from pipeline outputs.
    """
    return {
        "openalex_id":        paper_meta.openalex_id,
        "title":              paper_meta.title,
        "year":               paper_meta.year,
        "doi":                paper_meta.doi,
        "pdf_url":            paper_meta.pdf_url,
        "authors":            paper_meta.authors,
        "concepts":           paper_meta.concepts,
        "abstract":           segment.abstract,
        "extraction_method":  segment.extraction_method,
        "extraction_model":   "llama3.1:8b",
        "extracted_at":       datetime.now(timezone.utc).isoformat(),
        "methodology":        extraction_result.data,
        "extraction_success": extraction_result.success,
        "latency_seconds":    extraction_result.latency_seconds,
    }


def ensure_indexes(collection: Collection) -> None:
    """Create indexes for common query patterns."""
    collection.create_index([("openalex_id", ASCENDING)], unique=True)
    collection.create_index([("year", ASCENDING)])
    collection.create_index([("methodology.study_type", ASCENDING)])
    log.info("MongoDB indexes ensured")


# ---------------------------------------------------------------------------
# DuckDB export
# ---------------------------------------------------------------------------

DUCKDB_SCHEMA = """
CREATE TABLE IF NOT EXISTS papers (
    openalex_id         VARCHAR PRIMARY KEY,
    title               VARCHAR,
    year                INTEGER,
    doi                 VARCHAR,
    authors             VARCHAR,
    concepts            VARCHAR,
    abstract            VARCHAR,
    study_type          VARCHAR,
    sample_size         INTEGER,
    countries           VARCHAR,
    trust_measure       VARCHAR,
    statistical_models  VARCHAR,
    key_variables       VARCHAR,
    data_sources        VARCHAR,
    extraction_confidence VARCHAR,
    extraction_method   VARCHAR,
    extraction_model    VARCHAR,
    extracted_at        VARCHAR
)
"""


def export_to_duckdb(collection: Collection, duckdb_path: Path) -> int:
    """
    Export all successfully extracted papers from MongoDB to DuckDB.
    Flattens list fields to comma-separated strings for portability.

    Returns number of rows written.
    """
    con = duckdb.connect(str(duckdb_path))
    con.execute(DUCKDB_SCHEMA)

    docs = list(collection.find({"extraction_success": True}))
    if not docs:
        log.warning("No successful extractions found in MongoDB — nothing to export")
        con.close()
        return 0

    rows = [_flatten_document(doc) for doc in docs]

    # Upsert via staging table
    con.execute("CREATE TEMP TABLE staging AS SELECT * FROM papers LIMIT 0")
    con.executemany(
        """
        INSERT INTO staging VALUES (
            ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
        )
        """,
        [_row_to_tuple(r) for r in rows],
    )
    con.execute("""
        INSERT OR REPLACE INTO papers
        SELECT * FROM staging
    """)
    con.execute("DROP TABLE staging")
    con.close()

    log.info(f"Exported {len(rows)} papers to DuckDB: {duckdb_path}")
    return len(rows)


def _flatten_document(doc: dict) -> dict:
    """Flatten a MongoDB document into a DuckDB-ready flat dict."""
    m = doc.get("methodology", {}) or {}
    return {
        "openalex_id":          doc.get("openalex_id", ""),
        "title":                doc.get("title", ""),
        "year":                 doc.get("year"),
        "doi":                  doc.get("doi"),
        "authors":              ", ".join(doc.get("authors", [])),
        "concepts":             ", ".join(doc.get("concepts", [])),
        "abstract":             doc.get("abstract"),
        "study_type":           m.get("study_type"),
        "sample_size":          m.get("sample_size"),
        "countries":            ", ".join(m.get("countries", []) or []),
        "trust_measure":        m.get("trust_measure"),
        "statistical_models":   ", ".join(m.get("statistical_models", []) or []),
        "key_variables":        ", ".join(m.get("key_variables", []) or []),
        "data_sources":         ", ".join(m.get("data_sources", []) or []),
        "extraction_confidence": m.get("extraction_confidence"),
        "extraction_method":    doc.get("extraction_method"),
        "extraction_model":     doc.get("extraction_model"),
        "extracted_at":         doc.get("extracted_at"),
    }


def _row_to_tuple(row: dict) -> tuple:
    """Convert flat dict to tuple matching DuckDB schema column order."""
    return (
        row["openalex_id"],
        row["title"],
        row["year"],
        row["doi"],
        row["authors"],
        row["concepts"],
        row["abstract"],
        row["study_type"],
        row["sample_size"],
        row["countries"],
        row["trust_measure"],
        row["statistical_models"],
        row["key_variables"],
        row["data_sources"],
        row["extraction_confidence"],
        row["extraction_method"],
        row["extraction_model"],
        row["extracted_at"],
    )