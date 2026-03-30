# config.py
from pathlib import Path
from dotenv import load_dotenv
import os

load_dotenv()

BASE_DIR   = Path(__file__).parent
DATA_DIR   = BASE_DIR / "data"
PDF_DIR    = DATA_DIR / "pdfs"
RESULTS_DIR = BASE_DIR / "results"
(BASE_DIR / "logs").mkdir(exist_ok=True)
DATA_DIR.mkdir(exist_ok=True)
PDF_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)

# Ollama
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")
OLLAMA_MODEL    = os.getenv("OLLAMA_MODEL", "llama3.1:8b")

# MongoDB
MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")
MONGO_DB  = os.getenv("MONGO_DB", "trustlens")

# DuckDB
DUCKDB_PATH = DATA_DIR / "trustlens.duckdb"

# Pipeline settings
OPENALEX_EMAIL  = os.getenv("OPENALEX_EMAIL", "")   # polite pool — faster API
MAX_PAPERS      = int(os.getenv("MAX_PAPERS", "50"))
PDF_TIMEOUT_SEC = int(os.getenv("PDF_TIMEOUT_SEC", "30"))