# AI-Based College Academic Assistant (NMAMIT)

An AI assistant for students of **NMAM Institute of Technology (NMAMIT), Nitte**. It answers academic questions
from official college documents (regulations, syllabus, examination and internship rules), keeps conversational
context, uses a tool, and creates and edits personalised study plans.

**Stack:** RAG · LangChain · LangGraph · Groq LLM · Sentence Transformers · FAISS · Streamlit

```
STUDENT → STREAMLIT UI → LANGGRAPH → QUESTION ANALYSIS
                                        ├─ study plan?  → STUDY PLANNER → PLAN REVIEW → END
                                        └─ question     → RAG RETRIEVER (FAISS) → LANGCHAIN PROMPT + CONTEXT
                                                          → LLM (Groq) → RESPONSE REVIEW → FINAL ANSWER
```

## Team

| Member | Responsibility | Folder | Status |
|---|---|---|---|
| 1 | RAG / Knowledge Base: documents → chunks → embeddings → FAISS → retriever | `src/rag/` | ✅ Done |
| 2 | LangChain / Groq LLM / prompts / memory / calculator tool | `src/llm/` | ⏳ |
| 3 | LangGraph workflow + study planner | `src/graph/` | ⏳ |
| 4 | Streamlit UI, integration, testing | `app.py` | ⏳ |

## Quick start

```bash
pip install -r requirements.txt
```

The FAISS index is already committed in `vectorstore/`, so retrieval works straight away:

```bash
python scripts/query_kb.py "What is the minimum attendance required?"
```

```python
from src.rag import retrieve_context

result = retrieve_context("How long is Internship-II?")
result["found"]    # False -> say "not found in NMAMIT documents"
result["context"]  # text to put in the LLM prompt
result["sources"]  # citations to show in the UI
```

**Member 1 (RAG) docs, including the API for the other members:** [docs/MEMBER1_RAG_KNOWLEDGE_BASE.md](docs/MEMBER1_RAG_KNOWLEDGE_BASE.md)

## Project structure

```
├── data/
│   ├── sources.json             # list of documents and their metadata (title, batch, URL)
│   ├── raw/pdfs/                # official NMAMIT PDFs (from nitte.edu.in)
│   ├── raw/text/                # FAQ, internship and exam guideline summaries (with page references)
│   └── processed/               # chunk preview for inspection
├── src/rag/                     # Member 1: knowledge base
│   ├── config.py                #   all paths and settings
│   ├── loader.py                #   1. load PDFs / Markdown
│   ├── preprocess.py            #   2. clean text, tag course / section
│   ├── chunker.py               #   3. split into chunks
│   ├── embeddings.py            #   4. Sentence Transformers model
│   ├── vector_store.py          #   5. build / load FAISS
│   └── retriever.py             #   6. retrieve(question) → chunks  (public API)
├── scripts/
│   ├── build_index.py           # rebuild the index after changing documents
│   ├── query_kb.py              # try the retriever from the terminal
│   ├── evaluate_retrieval.py    # Hit@k / MRR / threshold calibration
│   └── download_documents.py    # re-download the official PDFs
├── tests/                       # pytest tests + evaluation questions
└── vectorstore/faiss_index/     # built FAISS index (committed)
```

## Common commands

```bash
python scripts/build_index.py              # rebuild the knowledge base (~7 min on CPU)
python scripts/build_index.py --dry-run    # only show chunking statistics
python scripts/evaluate_retrieval.py       # retrieval quality report
python -m pytest tests -v                  # run tests
```

## Data sources

All college documents are official public PDFs published by NMAMIT on [nitte.edu.in](https://nitte.edu.in/nmamit/).
The links are listed in `data/sources.json`. The FAQ and guideline files in `data/raw/text/` are summaries of those
PDFs, and every answer in them cites its PDF section and page.
