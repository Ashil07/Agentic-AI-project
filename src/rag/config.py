"""Central configuration for the RAG / knowledge-base module (Member 1).

Every path and tunable lives here so teammates never hard-code paths.
Values can be overridden with environment variables (see .env.example).
"""

import os
from pathlib import Path

# Project root = two levels above this file (src/rag/config.py -> project root)
PROJECT_ROOT = Path(__file__).resolve().parents[2]

try:  # load overrides from .env if python-dotenv is installed
    from dotenv import load_dotenv

    load_dotenv(PROJECT_ROOT / ".env")
except ImportError:
    pass

DATA_DIR = PROJECT_ROOT / "data"
RAW_PDF_DIR = DATA_DIR / "raw" / "pdfs"
RAW_TEXT_DIR = DATA_DIR / "raw" / "text"
SOURCES_FILE = DATA_DIR / "sources.json"
PROCESSED_DIR = DATA_DIR / "processed"
CHUNKS_PREVIEW_FILE = PROCESSED_DIR / "chunks_preview.jsonl"

VECTORSTORE_DIR = Path(os.getenv("VECTORSTORE_DIR", PROJECT_ROOT / "vectorstore" / "faiss_index"))
INDEX_MANIFEST_FILE = VECTORSTORE_DIR / "manifest.json"

# --- Embeddings -----------------------------------------------------------
# all-MiniLM-L6-v2: small (~90 MB), fast on CPU, 384-dim vectors.
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
EMBEDDING_DEVICE = os.getenv("EMBEDDING_DEVICE", "cpu")

# --- Chunking -------------------------------------------------------------
# MiniLM reads at most 256 tokens (~1000 characters), so chunks stay below that.
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 800))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 150))
MIN_CHUNK_CHARS = 80  # drop tiny fragments (page numbers, stray headings)

# --- Retrieval ------------------------------------------------------------
DEFAULT_TOP_K = int(os.getenv("RETRIEVER_TOP_K", 4))
# Cosine similarity (0..1) below which a chunk is treated as "not relevant".
# Calibrated with scripts/evaluate_retrieval.py; see docs/MEMBER1_RAG_KNOWLEDGE_BASE.md.
RELEVANCE_THRESHOLD = float(os.getenv("RELEVANCE_THRESHOLD", 0.40))
