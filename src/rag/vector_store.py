"""Step 5 of the pipeline: build, save and load the FAISS vector database."""

import json
import logging
import time
import warnings
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path

with warnings.catch_warnings():
    warnings.simplefilter("ignore", DeprecationWarning)
    from langchain_community.vectorstores import FAISS
    from langchain_community.vectorstores.utils import DistanceStrategy

from langchain_core.documents import Document

from . import config
from .chunker import split_documents
from .embeddings import get_embeddings
from .loader import load_all_documents
from .preprocess import preprocess_documents

logger = logging.getLogger(__name__)


def build_vector_store(save_dir: Path = config.VECTORSTORE_DIR) -> FAISS:
    """Run the full pipeline (load -> clean -> chunk -> embed -> index) and save it to disk."""
    start = time.time()
    raw_docs = load_all_documents()
    if not raw_docs:
        raise RuntimeError("No documents found. Add files to data/raw and list them in data/sources.json.")

    cleaned = preprocess_documents(raw_docs)
    chunks = split_documents(cleaned)
    logger.info("Pages/sections loaded: %d | cleaned segments: %d | chunks: %d", len(raw_docs), len(cleaned), len(chunks))

    logger.info("Embedding %d chunks with %s ...", len(chunks), config.EMBEDDING_MODEL_NAME)
    store = FAISS.from_documents(
        chunks,
        get_embeddings(),
        distance_strategy=DistanceStrategy.MAX_INNER_PRODUCT,  # = cosine, vectors are normalised
    )

    save_dir.mkdir(parents=True, exist_ok=True)
    store.save_local(str(save_dir))
    _write_manifest(save_dir, chunks, time.time() - start)
    _write_chunk_preview(chunks)
    logger.info("Saved FAISS index to %s (%.1fs)", save_dir, time.time() - start)
    load_vector_store.cache_clear()
    return store


@lru_cache(maxsize=1)
def load_vector_store(index_dir: Path = config.VECTORSTORE_DIR) -> FAISS:
    """Load the saved FAISS index (cached so it is read from disk only once)."""
    if not (Path(index_dir) / "index.faiss").exists():
        raise FileNotFoundError(
            f"No FAISS index at {index_dir}. Build it first with:  python scripts/build_index.py"
        )
    # allow_dangerous_deserialization: the .pkl docstore was created by our own
    # build script, so it is safe to load. Never load an index from an unknown source.
    return FAISS.load_local(
        str(index_dir),
        get_embeddings(),
        allow_dangerous_deserialization=True,
        distance_strategy=DistanceStrategy.MAX_INNER_PRODUCT,
    )


def _write_manifest(save_dir: Path, chunks: list[Document], seconds: float) -> None:
    per_source: dict[str, int] = {}
    courses: dict[str, str] = {}  # section label -> course code, used for course-aware retrieval
    for c in chunks:
        per_source[c.metadata["source"]] = per_source.get(c.metadata["source"], 0) + 1
        if c.metadata.get("course_code"):
            courses[c.metadata["section"]] = c.metadata["course_code"]
    manifest = {
        "built_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "embedding_model": config.EMBEDDING_MODEL_NAME,
        "chunk_size": config.CHUNK_SIZE,
        "chunk_overlap": config.CHUNK_OVERLAP,
        "total_chunks": len(chunks),
        "chunks_per_source": per_source,
        "build_seconds": round(seconds, 1),
        "courses": dict(sorted(courses.items())),
    }
    (save_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def _write_chunk_preview(chunks: list[Document], n: int = 300) -> None:
    """Save a sample of chunks as JSONL so the team can inspect what was indexed."""
    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    step = max(1, len(chunks) // n)
    with open(config.CHUNKS_PREVIEW_FILE, "w", encoding="utf-8") as f:
        for c in chunks[::step]:
            f.write(json.dumps({"text": c.page_content, "metadata": c.metadata}, ensure_ascii=False) + "\n")
