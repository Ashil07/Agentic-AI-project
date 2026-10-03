# Member 1: RAG / Knowledge Base

**Main deliverable:** Question → relevant NMAMIT document chunks.

This module turns NMAMIT's official documents into a searchable knowledge base and provides a single function,
`retrieve(question)`, that the LLM (Member 2), the LangGraph workflow (Member 3) and the UI (Member 4) call.

---

## 1. What was done (checklist from the team plan)

| Task | Done | Where |
|---|---|---|
| Collect college documents: syllabus, academic regulations, exam guidelines, internship guidelines, student FAQs | ✅ | `data/raw/`, `data/sources.json` |
| Load and preprocess documents | ✅ | `src/rag/loader.py`, `src/rag/preprocess.py` |
| Split documents into chunks | ✅ | `src/rag/chunker.py` |
| Generate embeddings with Sentence Transformers | ✅ | `src/rag/embeddings.py` |
| Create and populate the FAISS vector database | ✅ | `src/rag/vector_store.py`, `vectorstore/faiss_index/` |
| Build and test the retriever | ✅ | `src/rag/retriever.py`, `tests/`, `scripts/evaluate_retrieval.py` |
| Expose a simple retrieval function for other members | ✅ | `from src.rag import retrieve, retrieve_context` |

## 2. Knowledge base contents

| Document | Type | Applies to | Pages / sections | Chunks |
|---|---|---|---|---|
| B.Tech CSE Regulations & Curriculum 2023 | Regulations + full CSE syllabus | B.Tech, admitted 2023-24 onward | 408 pages | 1535 |
| B.Tech First Year Regulations, Scheme & Syllabus 2025 (Draft) | Regulations + first-year syllabus | B.Tech first year, 2025 | 173 pages | 624 |
| NMAMIT Regulations 2024-25 (160 credits) | Regulations | B.E., admitted 2021-22 onward | 28 pages | 90 |
| CSE VII & VIII Semester College Calendar & Syllabus 2024-25 | Syllabus + calendar + scholarships | B.E. CSE VII/VIII semester | 136 pages | 448 |
| NMAMIT Student FAQ | FAQ (compiled, with page references) | All UG students | 29 Q&As | 30 |
| NMAMIT Internship Guidelines (summary) | Internship guidelines | B.Tech 2023-24 onward | 12 sections | 14 |
| NMAMIT Examination Guidelines (summary) | Exam guidelines | B.Tech + B.E. | 9 sections | 10 |
| **Total** | | | **~795 pages/sections** | **2751** |

- The 4 PDFs are official documents from `nitte.edu.in` (links in `data/sources.json`).
- The 3 Markdown files are hand-written summaries of those PDFs in question-and-answer form. Every answer cites its
  PDF section and page. They exist because students ask short questions ("is internship compulsory?") that match a
  short FAQ entry much better than a long regulation paragraph.

**Adding a new document:** put the file in `data/raw/pdfs/` or `data/raw/text/`, add an entry to
`data/sources.json`, then run `python scripts/build_index.py`.

## 3. How the pipeline works

```
data/sources.json
      │
      ▼
1. LOAD        loader.py       PDF → one Document per page (pypdf);  Markdown → one Document per heading
      │
      ▼
2. CLEAN       preprocess.py   • remove headers/footers repeated on ≥40% of pages
                               • remove page numbers, fix broken characters (� → ' or -), join hyphenated words
                               • detect "Course Code: 21CS701" → tag text with "Course: Big Data Analytics (21CS701)"
                               • detect "9. ATTENDANCE REQUIREMENT", "APPENDIX - B" → tag text with its section
      │
      ▼
3. CHUNK       chunker.py      RecursiveCharacterTextSplitter, 800 chars, 150 overlap
                               every chunk starts with a header: [Document title | Course/Section | p.31]
      │
      ▼
4. EMBED       embeddings.py   sentence-transformers/all-MiniLM-L6-v2 → 384-dim normalised vectors
      │
      ▼
5. INDEX       vector_store.py FAISS (inner product on normalised vectors = cosine similarity), saved to disk
      │                         + manifest.json listing the 195 detected courses
      ▼
6. RETRIEVE    retriever.py    • if the question names a course ("Theory of Computation", "CS3003-1"),
                                 search that course's chunks first
                               • semantic search → drop score < threshold → de-duplicate → top k
```

### Design decisions (useful for the viva)

| Decision | Why |
|---|---|
| **all-MiniLM-L6-v2** | Small (~90 MB), fast on a laptop CPU, good semantic quality, and the standard Sentence Transformers starter model. |
| **Chunk size 800, overlap 150** | MiniLM reads at most 256 tokens (~1000 characters), so longer chunks would be cut off. The overlap keeps a rule from being split across two chunks without context. |
| **Context header in every chunk** | A syllabus chunk such as "UNIT-III: 8086 interrupts…" does not say which course it belongs to. Adding `Course: Microprocessor And Embedded Systems (CS3005-1)` lets both the retriever and the LLM know. |
| **Cosine similarity with a threshold** | Scores between 0 and 1 are easy to explain. If the best score is below the threshold, `retrieve()` returns `[]`, and the assistant says the information was not found instead of guessing (section 7 of the team plan). |
| **Course-aware retrieval** | All syllabus pages look alike (UNIT-I… Course Outcomes… Textbooks), so MiniLM sometimes ranked the wrong course first. If the question names one of the 195 detected courses (by title or code), that course's chunks are searched first. This raised MRR from 0.84 to 0.91 and fixed the Theory of Computation miss. |
| **De-duplication** | The B.E. regulation text is printed in two PDFs (the regulations PDF and the VII/VIII semester calendar). De-duplication stops the same paragraph from taking up two of the k slots. |
| **`applies_to` metadata** | NMAMIT has two regulation sets (B.E. 2021 batch and B.Tech 2023 batch) with some different rules, for example the CGPA-to-class conversion. Each chunk says which batch it applies to, so the LLM can tell them apart. |
| **Local FAISS, index committed to Git** | No cloud database is needed (the team plan says not to overbuild). Teammates can use the retriever without a ~7-minute rebuild. |

## 4. API for the other members

```python
from src.rag import retrieve, retrieve_context, format_context, get_retriever
```

### `retrieve(question, k=4, min_score=None, doc_types=None) -> list[RetrievedChunk]`

```python
chunks = retrieve("What is the minimum attendance required?")
for c in chunks:
    print(c.score, c.citation)
    # 0.68  NMAMIT Regulations 2024-25 (160 credits) - Section 6 Attendance Requirement (p.10)
```

`RetrievedChunk` fields: `text`, `score` (0-1), `source`, `title`, `page`, `section`, `doc_type`, `applies_to`, `url`,
plus the `citation` property and `to_dict()`.

- Returns an **empty list** when nothing is relevant (best score < `RELEVANCE_THRESHOLD`).
- `doc_types=["syllabus"]` limits results to one kind of document. Types: `regulations_and_syllabus`, `regulations`,
  `syllabus`, `faq`, `internship_guidelines`, `examination_guidelines`.
- `min_score=0` turns the threshold off.

### `retrieve_context(question, k=4) -> dict` (easiest for Members 2, 3 and 4)

```python
result = retrieve_context("How long is Internship-II?")
# {
#   "question": "How long is Internship-II?",
#   "found": True,
#   "context": "[Source 1] NMAMIT Internship Guidelines (summary) - ... (applies to: ...)\n<text>\n\n[Source 2] ...",
#   "sources": [{"citation": "...", "title": "...", "page": 33, "score": 0.72, "url": "...", ...}, ...]
# }
```

### `get_retriever(k=4, search_type="similarity")` (LangChain retriever for LCEL chains)

```python
retriever = get_retriever(k=4)            # or search_type="mmr" for more diverse results
docs = retriever.invoke("What is CGPA?")  # list[langchain_core.documents.Document]
```

This does **not** apply the relevance threshold. Use `retrieve()` / `retrieve_context()` when you need unknown-question handling.

### Suggested use by Member 2 (LLM answer with unknown-question handling)

```python
from src.rag import retrieve_context

result = retrieve_context(question)
if not result["found"]:
    answer = "I couldn't find this in the NMAMIT documents I have access to. Please check with your department or the academic section."
else:
    prompt = f"""You are the NMAMIT academic assistant. Answer ONLY from the context below.
If the context does not contain the answer, say it was not found in the NMAMIT documents.
If rules differ between batches (see 'applies to'), mention which batch each rule is for.
Cite sources as [Source N].

Context:
{result['context']}

Question: {question}"""
    answer = llm.invoke(prompt)
```

### Suggested use by Member 3 (LangGraph retrieval node)

```python
from src.rag import retrieve_context

def retrieval_node(state: dict) -> dict:
    result = retrieve_context(state["question"])
    return {"context": result["context"], "sources": result["sources"], "found": result["found"]}
```

### Suggested use by Member 4 (Streamlit, showing sources)

```python
with st.expander("Sources"):
    for s in result["sources"]:
        st.markdown(f"- **{s['citation']}** (relevance {s['score']:.2f})")
```

The first call loads the embedding model and the index (a few seconds). After that they are cached and each query takes
about 50 ms. In Streamlit, call `load_vector_store()` once at startup (for example inside `@st.cache_resource`) so the
first question is fast.

## 5. Evaluation results

Run with `python scripts/evaluate_retrieval.py` (40 NMAMIT questions, 8 unrelated questions, 5 college questions whose answer is not in the documents).

| Metric | Result |
|---|---|
| Hit@4 (an expected fact is in the top 4 chunks) | **100%** (40/40) |
| MRR (how high the first correct chunk ranks; 1.0 = always first) | **0.906** |
| Unrelated questions rejected (FIFA, recipes, weather…) | **8/8 (100%)** |
| NMAMIT questions wrongly rejected by the threshold | **0/40** |
| Top-1 similarity, NMAMIT questions | min 0.376 · avg 0.648 · max 0.807 |
| Top-1 similarity, unrelated questions | min 0.170 · avg 0.249 · max 0.327 |
| Top-1 similarity, college questions not in the documents | 0.434-0.606 |

**Threshold = 0.40.** Every unrelated question scores at most 0.33 and almost every NMAMIT question scores at least 0.42.
The single exception, a question naming a course, is protected by course-aware retrieval.

**How the results improved during development:**

| Version | Hit@4 | MRR | Change |
|---|---|---|---|
| v1 | 90% | 0.838 | baseline |
| v2 | 100% | 0.906 | Internship-I FAQ split into smaller sections, two FAQ entries added, course-aware retrieval |

## 6. Limitations / future improvements

- The CSE syllabus is the only branch syllabus included. Other branches' syllabus PDFs can be added the same way.
- Tables in PDFs (scheme of teaching, grade tables) lose their layout when converted to text. The FAQ files restate the
  most important tables in plain sentences.
- College facts that are not in these documents (fees, hostel, principal, events, placements) are not covered. The
  threshold catches clearly unrelated questions. For questions that are about the college but not covered, the LLM
  prompt must still say "not found" when the context doesn't contain the answer.
- Possible upgrades if time allows: hybrid search (BM25 + vectors) for exact course codes, a stronger embedding model
  (e.g. `BAAI/bge-small-en-v1.5`), or a cross-encoder re-ranker.

## 7. Viva quick answers

- **What is RAG?** Retrieval-Augmented Generation. First retrieve the relevant document chunks, then give them to the
  LLM as context, so its answer is grounded in NMAMIT's real rules instead of the LLM's general knowledge.
- **Why chunk?** LLMs and embedding models have input limits, and small focused chunks retrieve more precisely than
  whole documents.
- **What is an embedding?** A vector (384 numbers) that represents the meaning of a text. Similar meanings give
  vectors that point in similar directions, so their cosine similarity is high.
- **What does FAISS do?** It quickly finds the stored vectors closest to the question vector (here an exact search
  over 2,751 vectors).
- **How do you handle questions the documents can't answer?** If the best similarity is below the threshold,
  `retrieve()` returns nothing and the assistant says the information wasn't found.
