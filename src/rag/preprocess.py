"""Step 2 of the pipeline: clean raw PDF text and tag it with its section.

Text extracted from PDFs is noisy: repeated page headers, page numbers,
broken characters and lots of blank lines. This module fixes that, and
also tracks which course (e.g. "BIG DATA ANALYTICS (21CS701)") or which
regulation section (e.g. "9. ATTENDANCE REQUIREMENT") each piece of text
belongs to. That label is later added to every chunk, which improves
retrieval and lets the answer cite its source precisely.
"""

import re
from collections import Counter, defaultdict

from langchain_core.documents import Document

# --- Character-level cleaning ----------------------------------------------

_REPLACEMENTS = {
    "‘": "'", "’": "'", "“": '"', "”": '"',
    "–": "-", "—": "-", "•": "-", "": "-", " ": " ",
}
_PAGE_NUMBER_LINE = re.compile(r"^\s*(page\s*\|?\s*)?\d{1,3}\s*$", re.IGNORECASE)


def clean_text(text: str) -> str:
    """Normalise characters and whitespace in one page of text."""
    for bad, good in _REPLACEMENTS.items():
        text = text.replace(bad, good)
    # U+FFFD is pypdf's "unknown character". Between letters it is almost
    # always an apostrophe (student's); between spaces a dash (2023 - 2024).
    text = re.sub(r"(\w)�(\w)", r"\1'\2", text)
    text = re.sub(r"\s�\s", " - ", text)
    text = text.replace("�", "")
    # Join words hyphenated across a line break: "exami-\nnation" -> "examination"
    text = re.sub(r"([a-z])-\n([a-z])", r"\1\2", text)

    lines = []
    for line in text.splitlines():
        line = re.sub(r"[ \t]+", " ", line).strip()
        if _PAGE_NUMBER_LINE.match(line):
            continue
        lines.append(line)
    text = "\n".join(lines)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def find_repeated_lines(pages: list[str], min_fraction: float = 0.4, min_len: int = 8) -> set[str]:
    """Find header/footer lines that repeat on many pages of one document."""
    if len(pages) < 5:
        return set()
    counts = Counter()
    for page in pages:
        counts.update({line.strip() for line in page.splitlines() if len(line.strip()) >= min_len})
    return {line for line, n in counts.items() if n / len(pages) >= min_fraction}


# --- Section / course detection --------------------------------------------

# "Course Code : 21CS701" or "Course Code: CS3005-1"
_COURSE_CODE = re.compile(r"Course\s*Code\s*:\s*([A-Z0-9]{2,}(?:\s*[-]\s*\d)?)", re.IGNORECASE)
# "9. ATTENDANCE REQUIREMENT:" / "11.4 Termination from the programme" style headings in all caps
_SECTION_HEADING = re.compile(r"^(\d{1,2}(?:\.\d{1,2})?)\.?\s+([A-Z][A-Z &/,()'\-]{4,80}?)\s*:?\s*$", re.MULTILINE)
# "APPENDIX - A" / "APPENDIX-B" (appendices follow the last numbered section)
_APPENDIX_HEADING = re.compile(r"^(APPENDIX\s*-?\s*[A-Z0-9]{1,2})\b[ :]*([^\n]{0,60})$", re.MULTILINE)


def _looks_like_title(line: str) -> bool:
    letters = [c for c in line if c.isalpha()]
    if not (4 <= len(line) <= 90) or len(letters) < 4:
        return False
    return sum(c.isupper() for c in letters) / len(letters) > 0.7


def _find_markers(text: str) -> list[tuple[int, str, str | None]]:
    """Return (position, label, course_code) for each course/section start on a page."""
    markers = []
    for m in _COURSE_CODE.finditer(text):
        before = [l for l in text[: m.start()].rstrip().splitlines() if l.strip()]
        # The title is usually the line just above "Course Code", sometimes with
        # a note such as "(For CV Branch)" in between.
        title_line = next((l for l in reversed(before[-3:]) if _looks_like_title(l) and not l.startswith("(")), "")
        code = re.sub(r"\s+", "", m.group(1)).upper()
        if title_line:
            pos = text.rfind(title_line, 0, m.start())
            markers.append((pos, f"Course: {title_line.title()} ({code})", code))
        else:
            markers.append((m.start(), f"Course code {code}", code))
    for m in _SECTION_HEADING.finditer(text):
        markers.append((m.start(), f"Section {m.group(1)} {m.group(2).strip().title()}", None))
    for m in _APPENDIX_HEADING.finditer(text):
        label = re.sub(r"\s*-?\s*", "", m.group(1)[len("APPENDIX"):])
        markers.append((m.start(), f"Appendix {label} {m.group(2).strip().title()}".strip(), None))
    return sorted(markers, key=lambda x: x[0])


def preprocess_pdf_pages(pages: list[Document]) -> list[Document]:
    """Clean every page of ONE PDF and split pages at course/section boundaries.

    Returns Documents whose metadata has ``section`` (and ``course_code`` for
    syllabus text), carried forward from page to page until a new heading.
    """
    raw = [p.page_content for p in pages]
    repeated = find_repeated_lines(raw)

    out: list[Document] = []
    current_section, current_code = "", None
    for page in pages:
        text = "\n".join(l for l in page.page_content.splitlines() if l.strip() not in repeated)
        text = clean_text(text)
        if len(text) < 40:  # blank or image-only page
            continue

        # Cut the page at every marker; text before the first marker belongs
        # to whatever section was active at the end of the previous page.
        segments = []  # (start, end, section, course_code)
        prev = 0
        for pos, label, code in _find_markers(text):
            if pos > prev:
                segments.append((prev, pos, current_section, current_code))
                prev = pos
            current_section, current_code = label, code
        segments.append((prev, len(text), current_section, current_code))

        for start, end, section, code in segments:
            segment = text[start:end].strip()
            if len(segment) < 40:
                continue
            meta = {**page.metadata, "section": section}
            if code:
                meta["course_code"] = code
            out.append(Document(page_content=segment, metadata=meta))
    return out


def preprocess_documents(docs: list[Document]) -> list[Document]:
    """Clean all loaded documents. PDFs are processed per file; Markdown is already clean."""
    by_source: dict[str, list[Document]] = defaultdict(list)
    markdown: list[Document] = []
    for d in docs:
        if d.metadata.get("page") is None:
            markdown.append(Document(page_content=clean_text(d.page_content), metadata=d.metadata))
        else:
            by_source[d.metadata["source"]].append(d)

    cleaned: list[Document] = []
    for pages in by_source.values():
        cleaned.extend(preprocess_pdf_pages(pages))
    cleaned.extend(markdown)
    return cleaned
