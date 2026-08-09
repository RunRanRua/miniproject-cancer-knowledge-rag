from pathlib import Path


# =========================
# Project
# =========================

BASE_DIR = Path(__file__).resolve().parent


# =========================
# Data
# =========================

DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
NCI_DATA_DIR = RAW_DATA_DIR / "nci"
MANIFEST_FILE = DATA_DIR / "manifest.json"

# =========================
# Vector Store
# =========================

VECTOR_STORE_DIR = BASE_DIR / "vector_store"


# =========================
# Models
# =========================

LLM_PROVIDER = "ollama"
OLLAMA_BASE_URL = "http://localhost:11434"
OLLAMA_MODEL = "minimax-m3:cloud"

EMBEDDING_MODEL = "..."