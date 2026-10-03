"""Step 6: the retriever, the main thing other members use.

Contract (as defined in the team plan): question in -> relevant chunks out.

    from src.rag import retrieve, format_context

    chunks = retrieve("What is the minimum attendance required?")
    if not chunks:
        ...  # nothing relevant in the NMAMIT documents -> say "not found"
    context = format_context(chunks)   # paste into the LLM prompt
"""

import json
import re
from dataclasses import asdict, dataclass
from functools import lru_cache

from langchain_core.vectorstores import VectorStoreRetriever

from . import config
from .vector_store import load_vector_store


@dataclass
class RetrievedChunk:
    """One relevant piece of an NMAMIT document."""

    text: str  # chunk text, starting with a [title | section | page] header
    score: float  # cosine similarity, 0..1 (higher = more relevant)
    source: str  # file name, e.g. NMAMIT_Rules_and_Regulations_160_credits.pdf
    title: str  # human-readable document title
    page: int | None  # PDF page number (None for Markdown/FAQ documents)
    section: str  # course or regulation section, e.g. "Section 9 Attendance Requirement"
    doc_type: str  # regulations | syllabus | faq | internship_guidelines | ...
    applies_to: str  # which students/batch the document applies to
    url: str  # official download link, if any

    @property
    def citation(self) -> str:
        """Short reference for showing sources in the UI, e.g. 'NMAMIT Student FAQ - Attendance'."""
        cite = self.title
        if self.section:
            cite += f" - {self.section}"
        if self.page:
            cite += f" (p.{self.page})"
        return cite

    def to_dict(self) -> dict:
        return {**asdict(self), "citation": self.citation}


def _dedupe_key(text: str) -> str:
    body = text.split("\n", 1)[-1]  # ignore the [header] line
    return re.sub(r"[^a-z0-9]", "", body.lower())[:250]


# --- Course-aware retrieval --------------------------------------------------
# Syllabus pages all look alike ("UNIT-I ... Course Outcomes ... TEXTBOOKS"), so
# the embedding model can confuse courses. If the question names a course
# ("Theory of Computation", "CS3003-1", "21CS701"), that course's chunks are
# searched first.


def _normalise(text: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", text.lower().replace("&", " and ")))


@lru_cache(maxsize=1)
def _course_lookup() -> tuple[tuple[str, str, str], ...]:
    """(normalised title, normalised code, section label) for every course in the index."""
    try:
        courses = json.loads(config.INDEX_MANIFEST_FILE.read_text(encoding="utf-8")).get("courses", {})
    except (OSError, ValueError):
        return ()
    lookup = []
    for section, code in courses.items():
        m = re.match(r"Course: (.+) \((.+)\)$", section)
        lookup.append((_normalise(m.group(1)) if m else "", _normalise(code), section))
    return tuple(lookup)


def courses_mentioned(question: str) -> list[str]:
    """Section labels of courses named in the question (longest title match wins)."""
    q = f" {_normalise(question)} "
    matches = []
    for title, code, section in _course_lookup():
        if len(title) >= 8 and f" {title} " in q:
            matches.append((len(title), section))
        elif code and f" {code} " in q:
            matches.append((100, section))  # an exact course code is the strongest signal
    if not matches:
        return []
    best = max(length for length, _ in matches)  # "machine learning lab" beats "machine learning"
    return [section for length, section in matches if length == best][:3]


def retrieve(
    question: str,
    k: int = config.DEFAULT_TOP_K,
    min_score: float | None = None,
    doc_types: list[str] | None = None,
) -> list[RetrievedChunk]:
    """Return up to ``k`` chunks relevant to ``question``, best first.

    Args:
        question: the student's question.
        k: maximum number of chunks to return.
        min_score: drop chunks with cosine similarity below this value.
            Defaults to config.RELEVANCE_THRESHOLD. Pass 0 to always get k results.
        doc_types: optionally restrict to some document types, e.g. ["syllabus"].

    Returns:
        A list of RetrievedChunk, best first (chunks of a course named in the
        question are placed first). An EMPTY list means the knowledge base has
        nothing relevant, and the assistant should say the information was not
        found in the NMAMIT documents instead of guessing.
    """
    question = (question or "").strip()
    if not question:
        return []
    threshold = config.RELEVANCE_THRESHOLD if min_score is None else min_score

    store = load_vector_store()
    candidates = []  # (doc, score, named_course)
    # 1. Chunks of any course the question names come first. The question
    #    names a course we have, so these skip the relevance threshold.
    for section in courses_mentioned(question):
        hits = store.similarity_search_with_score(question, k=k, filter={"section": section}, fetch_k=store.index.ntotal)
        candidates += [(doc, score, True) for doc, score in hits]
    # 2. Then normal semantic search. Over-fetch, then filter, de-duplicate and cut to k.
    candidates += [(doc, score, False) for doc, score in store.similarity_search_with_score(question, k=max(k * 4, 20))]

    results: list[RetrievedChunk] = []
    seen: set[str] = set()
    for doc, score, named_course in candidates:
        meta = doc.metadata
        if score < threshold and not named_course:
            continue
        if doc_types and meta.get("doc_type") not in doc_types:
            continue
        key = _dedupe_key(doc.page_content)
        if key in seen:  # the same regulation text appears in more than one PDF
            continue
        seen.add(key)
        results.append(
            RetrievedChunk(
                text=doc.page_content,
                score=round(float(score), 4),
                source=meta.get("source", ""),
                title=meta.get("title", ""),
                page=meta.get("page"),
                section=meta.get("section", ""),
                doc_type=meta.get("doc_type", ""),
                applies_to=meta.get("applies_to", ""),
                url=meta.get("url", ""),
            )
        )
        if len(results) == k:
            break
    return results


def has_relevant_context(chunks: list[RetrievedChunk]) -> bool:
    """True if retrieval found something relevant enough to answer from."""
    return len(chunks) > 0


def format_context(chunks: list[RetrievedChunk]) -> str:
    """Format chunks as a numbered context block for an LLM prompt.

    Example output:
        [Source 1] NMAMIT Student FAQ - Attendance > What is the minimum ... (applies to: ...)
        <chunk text>
    """
    if not chunks:
        return "NO RELEVANT NMAMIT DOCUMENTS FOUND."
    blocks = []
    for i, c in enumerate(chunks, start=1):
        body = c.text.split("\n", 1)[-1]  # the citation line already carries the header info
        blocks.append(f"[Source {i}] {c.citation} (applies to: {c.applies_to})\n{body}")
    return "\n\n".join(blocks)


def retrieve_context(question: str, k: int = config.DEFAULT_TOP_K) -> dict:
    """Do everything in one call, for LangGraph nodes / the UI.

    Returns:
        {"question": str, "found": bool, "context": str,
         "sources": [ {citation, title, page, score, url, ...}, ... ]}
    """
    chunks = retrieve(question, k=k)
    return {
        "question": question,
        "found": has_relevant_context(chunks),
        "context": format_context(chunks),
        "sources": [c.to_dict() for c in chunks],
    }


def get_retriever(k: int = config.DEFAULT_TOP_K, search_type: str = "similarity") -> VectorStoreRetriever:
    """Return a standard LangChain retriever (for use inside LCEL chains).

    search_type: "similarity" (default) or "mmr" (more diverse results).
    Note: this plain LangChain retriever does not apply the relevance
    threshold. Use ``retrieve()`` when you need unknown-question handling.
    """
    search_kwargs = {"k": k}
    if search_type == "mmr":
        search_kwargs["fetch_k"] = max(k * 5, 20)
    return load_vector_store().as_retriever(search_type=search_type, search_kwargs=search_kwargs)
