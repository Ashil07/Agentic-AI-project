"""Tests for the RAG / knowledge-base module.

Run from the project root:
    python -m pytest tests -v

The preprocessing tests run without the index; the retrieval tests need
the FAISS index (python scripts/build_index.py) and are skipped otherwise.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from langchain_core.documents import Document  # noqa: E402

from src.rag import config  # noqa: E402
from src.rag.chunker import split_documents  # noqa: E402
from src.rag.preprocess import clean_text, find_repeated_lines, preprocess_pdf_pages  # noqa: E402

INDEX_EXISTS = (config.VECTORSTORE_DIR / "index.faiss").exists()
needs_index = pytest.mark.skipif(not INDEX_EXISTS, reason="build the index first: python scripts/build_index.py")


# ---------------------------------------------------------------- preprocessing


def test_clean_text_fixes_broken_characters_and_page_numbers():
    raw = "student�s grade\n\n\n\nPage | 12\n2023 � 2024\n   lots   of   space  "
    out = clean_text(raw)
    assert "student's" in out
    assert "2023 - 2024" in out
    assert "Page | 12" not in out
    assert "lots of space" in out
    assert "\n\n\n" not in out


def test_repeated_header_lines_are_detected():
    pages = [f"Regulations and curriculum for B. Tech.\nbody text {i}" for i in range(10)]
    assert "Regulations and curriculum for B. Tech." in find_repeated_lines(pages)


def test_course_title_and_code_are_tagged():
    meta = {"source": "x.pdf", "title": "Test", "doc_type": "syllabus", "applies_to": "", "url": ""}
    page = Document(
        page_content="Some text from the previous course " * 3
        + "\nBIG DATA ANALYTICS\nCourse Code : 21CS701 CIE Marks : 50\nUNIT - I Introduction to big data.",
        metadata={**meta, "page": 44},
    )
    segments = preprocess_pdf_pages([page])
    tagged = [s for s in segments if s.metadata.get("course_code") == "21CS701"]
    assert tagged and tagged[0].metadata["section"] == "Course: Big Data Analytics (21CS701)"
    assert "Introduction to big data" in tagged[0].page_content


def test_chunks_have_header_and_respect_size():
    meta = {"source": "x.pdf", "title": "Doc", "doc_type": "regulations", "applies_to": "", "url": "", "page": 3,
            "section": "Section 9 Attendance"}
    doc = Document(page_content=("Attendance rule sentence. " * 200), metadata=meta)
    chunks = split_documents([doc], chunk_size=500, chunk_overlap=50)
    assert len(chunks) > 1
    for c in chunks:
        assert c.page_content.startswith("[Doc | Section 9 Attendance | p.3]")
        assert len(c.page_content) <= 500 + 60  # chunk + header line
        assert c.metadata["chunk_id"]


# -------------------------------------------------------------------- retrieval


@needs_index
def test_retrieve_returns_ranked_relevant_chunks():
    from src.rag import retrieve

    chunks = retrieve("What is the minimum attendance required?", k=4)
    assert 1 <= len(chunks) <= 4
    assert any("85%" in c.text for c in chunks)
    scores = [c.score for c in chunks]
    assert scores == sorted(scores, reverse=True)
    assert all(c.citation for c in chunks)


@needs_index
def test_unrelated_question_returns_nothing():
    from src.rag import retrieve, retrieve_context

    assert retrieve("Who won the FIFA world cup in 2022?") == []
    result = retrieve_context("How do I bake a chocolate cake?")
    assert result["found"] is False
    assert result["sources"] == []


@needs_index
def test_retrieve_context_shape():
    from src.rag import retrieve_context

    result = retrieve_context("How long is Internship-II?")
    assert result["found"] is True
    assert set(result) == {"question", "found", "context", "sources"}
    assert result["context"].startswith("[Source 1]")
    assert {"citation", "score", "title", "page"} <= set(result["sources"][0])


@needs_index
def test_doc_type_filter():
    from src.rag import retrieve

    chunks = retrieve("big data analytics syllabus", k=3, doc_types=["syllabus"])
    assert chunks and all(c.doc_type == "syllabus" for c in chunks)


@needs_index
def test_course_named_in_question_is_retrieved_first():
    from src.rag import retrieve
    from src.rag.retriever import courses_mentioned

    assert courses_mentioned("What units are in Theory of Computation?")
    assert courses_mentioned("syllabus of CS3003-1")
    assert courses_mentioned("What is the minimum attendance?") == []

    chunks = retrieve("What units are in Theory of Computation?", k=3)
    assert "Theory Of Computation" in chunks[0].section


@needs_index
def test_empty_question():
    from src.rag import retrieve

    assert retrieve("   ") == []


@needs_index
def test_langchain_retriever_interface():
    from src.rag import get_retriever

    docs = get_retriever(k=2).invoke("What is CGPA?")
    assert len(docs) == 2 and isinstance(docs[0], Document)
