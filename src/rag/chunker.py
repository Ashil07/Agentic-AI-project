"""Step 3 of the pipeline: split cleaned documents into chunks.

Each chunk starts with a short context header such as
``[B.Tech CSE Regulations & Curriculum 2023 | Section 9 Attendance Requirement | p.31]``
so both the embedding model and the LLM know where the text came from.
"""

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from . import config


def _context_header(meta: dict) -> str:
    parts = [meta["title"]]
    if meta.get("section"):
        parts.append(meta["section"])
    if meta.get("page"):
        parts.append(f"p.{meta['page']}")
    return "[" + " | ".join(parts) + "]"


def split_documents(
    docs: list[Document],
    chunk_size: int = config.CHUNK_SIZE,
    chunk_overlap: int = config.CHUNK_OVERLAP,
) -> list[Document]:
    """Split documents into overlapping chunks and give each a stable id."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        # Prefer to break at paragraphs, then lines, then sentences.
        separators=["\n\n", "\n", ". ", "; ", ", ", " ", ""],
    )

    chunks: list[Document] = []
    for doc in docs:
        for piece in splitter.split_text(doc.page_content):
            if len(piece.strip()) < config.MIN_CHUNK_CHARS:
                continue
            meta = dict(doc.metadata)
            meta["chunk_id"] = f"{meta['source']}::{meta.get('page') or meta.get('heading', '')}::{len(chunks)}"
            content = f"{_context_header(meta)}\n{piece.strip()}"
            chunks.append(Document(page_content=content, metadata=meta))
    return chunks
