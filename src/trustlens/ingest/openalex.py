"""
src/trustlens/ingest/openalex.py
=================================
Fetch paper metadata from the OpenAlex API.

Targets: open-access papers on social trust, filtered to those
with a reachable PDF URL (open_access.oa_url).

OpenAlex docs: https://docs.openalex.org
"""

import logging
import time
import requests
from dataclasses import dataclass

log = logging.getLogger(__name__)

OPENALEX_BASE = "https://api.openalex.org/works"

SEARCH_QUERY = (
    "social trust"
)

FILTERS = ",".join([
    "open_access.is_oa:true",
    "type:article",
    "from_publication_date:2010-01-01",
])


@dataclass
class PaperMeta:
    """Minimal metadata for a single paper."""
    openalex_id: str
    title: str
    year: int
    doi: str | None
    pdf_url: str | None
    abstract: str | None
    authors: list[str]
    concepts: list[str]


def fetch_papers(
    email: str,
    max_results: int = 50,
    query: str = SEARCH_QUERY,
) -> list[PaperMeta]:
    """
    Fetch open-access paper metadata from OpenAlex.

    Args:
        email:      Your email — adds you to the polite pool (faster rate limits)
        max_results: Maximum number of papers to return
        query:      Search query string

    Returns:
        List of PaperMeta objects with PDF URLs populated
    """
    papers = []
    page = 1
    per_page = min(max_results, 25)   # OpenAlex max per page is 200, but 25 is safe

    log.info(f"Fetching up to {max_results} papers — query: {query!r}")

    while len(papers) < max_results:
        params = {
            "search":     query,
            "filter":     FILTERS,
            "per-page":   per_page,
            "page":       page,
            "select":     "id,title,publication_year,doi,open_access,abstract_inverted_index,authorships,concepts",
            "mailto":     email,
        }

        try:
            resp = requests.get(OPENALEX_BASE, params=params, timeout=15)
            resp.raise_for_status()
        except requests.RequestException as e:
            log.error(f"OpenAlex request failed (page {page}): {e}")
            break

        data = resp.json()
        results = data.get("results", [])

        if not results:
            log.info("No more results from OpenAlex.")
            break

        for item in results:
            pdf_url = _extract_pdf_url(item)
            if not pdf_url:
                continue   # skip — no downloadable PDF

            paper = PaperMeta(
                openalex_id = item["id"],
                title       = item.get("title") or "",
                year        = item.get("publication_year") or 0,
                doi         = item.get("doi"),
                pdf_url     = pdf_url,
                abstract    = _reconstruct_abstract(item.get("abstract_inverted_index")),
                authors     = _extract_authors(item.get("authorships", [])),
                concepts    = _extract_concepts(item.get("concepts", [])),
            )
            papers.append(paper)
            log.debug(f"  Found: {paper.title[:60]} ({paper.year})")

            if len(papers) >= max_results:
                break

        log.info(f"Page {page} — collected {len(papers)}/{max_results} papers so far")
        page += 1
        time.sleep(0.5)   # polite rate limiting

    log.info(f"Fetch complete — {len(papers)} papers with PDF URLs")
    return papers


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _extract_pdf_url(item: dict) -> str | None:
    """Extract the best available open-access PDF URL."""
    oa = item.get("open_access", {})
    return oa.get("oa_url")


def _reconstruct_abstract(inverted_index: dict | None) -> str | None:
    """
    OpenAlex stores abstracts as inverted indexes {word: [positions]}.
    Reconstruct the original string.
    """
    if not inverted_index:
        return None
    try:
        max_pos = max(pos for positions in inverted_index.values() for pos in positions)
        words = [""] * (max_pos + 1)
        for word, positions in inverted_index.items():
            for pos in positions:
                words[pos] = word
        return " ".join(words).strip()
    except Exception:
        return None


def _extract_authors(authorships: list) -> list[str]:
    """Extract author display names."""
    authors = []
    for a in authorships:
        name = a.get("author", {}).get("display_name")
        if name:
            authors.append(name)
    return authors


def _extract_concepts(concepts: list) -> list[str]:
    """Extract top concept labels (score > 0.3)."""
    return [
        c["display_name"]
        for c in concepts
        if c.get("score", 0) > 0.3
    ]