"""Step 1 of the pipeline: load NMAMIT source documents.

Reads data/sources.json and returns one LangChain ``Document`` per PDF page
or per Markdown section, with metadata describing where it came from.
"""

import json
import logging
from pathlib import Path

from langchain_core.documents import Document
from langchain_text_splitters import MarkdownHeaderTextSplitter
from pypdf import PdfReader

from . import config

logger = logging.getLogger(__name__)


def load_source_manifest(path: Path = config.SOURCES_FILE) -> list[dict]:
    """Return the list of document entries from data/sources.json."""
    with open(path, encoding="utf-8") as f:
        return json.load(f)["documents"]


def _base_metadata(entry: dict) -> dict:
    return {
        "source": Path(entry["file"]).name,
        "title": entry["title"],
        "doc_type": entry["doc_type"],
        "applies_to": entry.get("applies_to", ""),
        "url": entry.get("url", ""),
    }


def load_pdf(path: Path, entry: dict) -> list[Document]:
    """Load a PDF as one Document per page (raw text, cleaned later)."""
    reader = PdfReader(str(path))
    docs = []
    for page_number, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        docs.append(Document(page_content=text, metadata={**_base_metadata(entry), "page": page_number}))
    return docs


def load_markdown(path: Path, entry: dict) -> list[Document]:
    """Load a Markdown file as one Document per heading section.

    Each FAQ question (### heading) becomes its own Document, so a question
    and its answer always stay together in one chunk.
    """
    text = path.read_text(encoding="utf-8")
    splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=[("#", "h1"), ("##", "h2"), ("###", "h3")],
        strip_headers=True,
    )
    docs = []
    for section in splitter.split_text(text):
        heading = section.metadata.get("h3") or section.metadata.get("h2") or section.metadata.get("h1", "")
        parent = section.metadata.get("h2", "") if section.metadata.get("h3") else ""
        docs.append(
            Document(
                page_content=section.page_content,
                metadata={
                    **_base_metadata(entry),
                    "page": None,
                    "section": f"{parent} > {heading}" if parent else heading,
                    "heading": heading,
                },
            )
        )
    return docs


def load_all_documents(raw_dir: Path = config.DATA_DIR / "raw") -> list[Document]:
    """Load every document listed in data/sources.json.

    Files that are listed but missing are skipped with a warning, so the
    index can still be built while documents are being collected.
    """
    docs: list[Document] = []
    for entry in load_source_manifest():
        path = raw_dir / entry["file"]
        if not path.exists():
            logger.warning("Listed in sources.json but missing on disk: %s", path)
            continue
        suffix = path.suffix.lower()
        if suffix == ".pdf":
            loaded = load_pdf(path, entry)
        elif suffix in {".md", ".txt"}:
            loaded = load_markdown(path, entry)
        else:
            logger.warning("Unsupported file type, skipping: %s", path)
            continue
        logger.info("Loaded %-60s %4d pages/sections", path.name, len(loaded))
        docs.extend(loaded)
    return docs
