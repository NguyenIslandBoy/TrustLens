# TrustLens

> A fully local, reproducible batch pipeline for extracting structured methodology metadata from open-access academic papers on social trust.

No external APIs for inference. No data leaves your machine.

---

## What it does

TrustLens fetches open-access papers from [OpenAlex](https://openalex.org), downloads the PDFs, isolates the methods section, and sends it to a local LLM (Llama 3.1 8B via Ollama) to extract structured methodology metadata. Results are stored in MongoDB for inspection and exported to DuckDB for versioned public release.

```
OpenAlex API → PDF download → text extraction → methods segmentation
     → local LLM extraction → MongoDB → DuckDB export
```

Extracted fields per paper:

| Field | Description |
|---|---|
| `study_type` | quantitative / qualitative / mixed |
| `sample_size` | number of participants or observations |
| `countries` | where the study was conducted |
| `trust_measure` | how trust was operationalised |
| `statistical_models` | e.g. OLS regression, SEM, multilevel modelling |
| `key_variables` | main independent and dependent variables |
| `data_sources` | datasets or surveys used |
| `extraction_confidence` | high / medium / low |

---

## Architecture

```
src/trustlens/
├── ingest/
│   ├── openalex.py       # fetch paper metadata from OpenAlex API
│   └── pdf_fetcher.py    # download and validate open-access PDFs
├── parse/
│   ├── extractor.py      # PDF → text (pdfplumber, page-aware)
│   └── segmenter.py      # isolate methods section via header patterns
├── llm/
│   ├── prompt.py         # system + user prompt templates
│   └── batch_runner.py   # LLM calls, JSON parsing, retry logic
├── store/
│   └── database.py       # MongoDB upsert + DuckDB export
└── pipeline.py           # end-to-end orchestrator
```

**Storage:**
- **MongoDB** - live operational store, one document per paper, inspectable via Compass
- **DuckDB** - flattened versioned export for analysis and public release

---

## Results

Pipeline validated on 50 open-access social science papers on social trust (March 2026).

| Metric | Value |
|---|---|
| Papers fetched | 50 |
| PDFs downloaded | 21 (42%) |
| LLM extraction success | 49/50 (98%) |
| MongoDB documents | 50 |
| DuckDB rows exported | 49 |

**Note on PDF download rate:** 29 papers were blocked by publisher paywalls (Sciencedirect, Wiley) despite being marked open-access by OpenAlex. This is a known limitation of OA metadata — Unpaywall API integration is the natural next step to improve coverage.

---

## Quickstart

### Prerequisites

- Python 3.11+
- [Ollama](https://ollama.com) running locally with `llama3.1:8b` pulled
- [MongoDB Community](https://www.mongodb.com/try/download/community) running on `localhost:27017`

```bash
ollama pull llama3.1:8b
```

### Install

```bash
git clone https://github.com/yourname/trustlens.git
cd trustlens
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # macOS/Linux
pip install -e .
```

### Configure

```bash
cp .env.example .env
# Edit .env and set your email for the OpenAlex polite pool
```

### Run

```bash
# Full pipeline - fetch, download, extract, store, export
python -m src.trustlens.pipeline

# Limit to 10 papers for a quick test
python -m src.trustlens.pipeline --max-papers 10

# Skip PDF download (reuse already-downloaded files)
python -m src.trustlens.pipeline --skip-download

# Ingestion only - no LLM calls
python -m src.trustlens.pipeline --skip-extraction --no-export
```

### Run tests

```bash
pytest tests/ -v
pytest tests/ -v --cov=src --cov-report=term-missing
```

---

## Configuration

All settings are in `.env`:

```dotenv
OLLAMA_BASE_URL=http://localhost:11434/v1
OLLAMA_MODEL=llama3.1:8b
MONGO_URI=mongodb://localhost:27017
MONGO_DB=trustlens
OPENALEX_EMAIL=your_email@example.com   # adds you to OpenAlex polite pool
MAX_PAPERS=50
PDF_TIMEOUT_SEC=30
```

---

## Design decisions

**Idempotent downloads** - PDFs already on disk are skipped. Safe to re-run the pipeline without re-downloading everything.

**PDF validation** - files are checked against the `%PDF` magic bytes. HTML error pages and paywalls returned by some publishers are rejected rather than passed to the extractor.

**Graceful fallback** - when no methods section header is found in a paper, the pipeline falls back to abstract + paper opening rather than failing. The `extraction_method` field (`methods_section` or `fallback`) is stored in MongoDB so you can filter by extraction quality downstream.

**Confidence guard** - extractions using the fallback path with no abstract are automatically downgraded from `high` to `low` confidence, flagging them for manual review.

**Never raises** - every stage returns a structured result object rather than propagating exceptions. The pipeline always runs to completion regardless of individual paper failures.

**Two-layer storage** - MongoDB for live inspection during a pipeline run, DuckDB for the versioned flat export that can be published or queried analytically.

---

## Security & data handling

All LLM inference is handled locally by Ollama. No paper text, queries, or responses are sent to external services. The OpenAlex API is used only to fetch public bibliographic metadata and open-access PDF URLs.

---

## Repo structure

```
trustlens/
├── src/trustlens/        # pipeline source code
├── tests/                # pytest suite
├── data/
│   ├── pdfs/             # downloaded PDFs (gitignored)
│   └── trustlens.duckdb  # DuckDB export (gitignored)
├── results/              # versioned JSON snapshots
├── logs/                 # pipeline logs (gitignored)
├── config.py             # settings and path configuration
├── pyproject.toml        # package definition
└── requirements.txt      # pinned dependencies
```

---

## Limitations

- Methods section detection is heuristic - papers with non-standard formatting fall back to abstract + opening text
- Scanned or image-based PDFs are not supported (no OCR)
- Sample size extraction can conflate study N with other numbers (e.g. number of estimates in a meta-analysis)
- Pipeline is sequential - no parallel PDF downloads or LLM calls
