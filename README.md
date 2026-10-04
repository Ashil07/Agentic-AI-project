AI-Based College Academic Assistant (NMAMIT)
An AI-powered academic assistant for students of NMAM Institute of Technology (NMAMIT), Nitte.
The assistant answers academic questions using official college documents, maintains conversational context, performs mathematical calculations, and creates and modifies personalised study plans.
Features
Academic Question Answering using RAG over official NMAMIT documents
Conversational Memory for follow-up questions
Calculator Tool for mathematical calculations
Study Plan Generation based on subjects, available study hours, and examination dates
Study Plan Modification based on changes in the student's schedule
Source Citations for retrieved college information
Hallucination Prevention by refusing to answer college-specific questions when relevant information is not available in the knowledge base
Streamlit Web Interface
Tech Stack
Python
Streamlit – User interface
LangGraph – Workflow and intent routing
LangChain – LLM integration, prompts, tools and memory
Groq – Large Language Model API
Sentence Transformers – Text embeddings
FAISS – Vector database for document retrieval
Pytest – Automated testing
System Architecture
```text
                    STUDENT
                       |
                       v
                STREAMLIT UI
                       |
                       v
              LANGGRAPH WORKFLOW
                       |
                INTENT ANALYSIS
                  /          \\
                 /            \\
                v              v
        ACADEMIC QUESTION   STUDY PLAN
                |              |
                v              v
          MEMBER 2 LLM     STUDY PLANNER
                |
        +-------+-------+
        |       |       |
        v       v       v
       RAG    MEMORY  CALCULATOR
        |
        v
   FAISS RETRIEVER
        |
        v
   COLLEGE CONTEXT
        |
        +-------+
                |
                v
            GROQ LLM
                |
                v
       FINAL ANSWER + SOURCES
```
Team
Member	Responsibility	Folder	Status
1	RAG / Knowledge Base: documents → chunks → embeddings → FAISS → retriever	`src/rag/`	Done
2	LangChain / Groq LLM / prompts / memory / calculator tool	`src/llm/`	Done
3	LangGraph workflow + study planner	`src/graph/`	Done
4	Streamlit UI, integration and testing	`app.py`	Done
Getting Started
1. Clone the repository
```bash
git clone https://github.com/Ashil07/Agentic-AI-project.git
cd Agentic-AI-project
```
2. Create a virtual environment
```bash
python -m venv .venv
```
Activate it on Windows PowerShell:
```powershell
.venv\Scripts\Activate.ps1
```
3. Install dependencies
```bash
python -m pip install -r requirements.txt
```
4. Configure the Groq API key
Create a `.env` file from the provided template:
```powershell
Copy-Item .env.example .env
```
Open `.env` and add your own Groq API key:
```text
GROQ_API_KEY=your_groq_api_key_here
```
Do not commit `.env` to GitHub.
The `.env.example` file is provided as a safe configuration template.
5. Run the application
```bash
streamlit run app.py
```
The Streamlit application will open in your browser.
Testing
Run the complete automated test suite using:
```bash
python -m pytest
```
The tests cover the RAG system and integrated academic assistant workflow, including:
Academic question answering
Unknown/out-of-context questions
Calculator functionality
Study plan generation
Study plan modification
RAG retrieval
API and workflow behaviour
Project Structure
```text
├── app.py                         # Streamlit application
├── requirements.txt               # Python dependencies
├── .env.example                   # Environment variable template
│
├── data/
│   ├── sources.json               # Document metadata and URLs
│   ├── raw/
│   │   ├── pdfs/                  # Official NMAMIT PDFs
│   │   └── text/                  # Processed guideline/FAQ content
│   └── processed/                 # Processed/chunked data
│
├── src/
│   ├── rag/                       # Member 1: RAG knowledge base
│   │   ├── config.py
│   │   ├── loader.py
│   │   ├── preprocess.py
│   │   ├── chunker.py
│   │   ├── embeddings.py
│   │   ├── vector_store.py
│   │   └── retriever.py
│   │
│   ├── llm/                       # Member 2: LLM integration
│   │   ├── groq_llm.py
│   │   ├── memory.py
│   │   └── calculator.py
│   │
│   └── graph/                     # Member 3: LangGraph + study planner
│       ├── agent_graph.py
│       └── study_planner.py
│
├── scripts/
│   ├── build_index.py             # Rebuild FAISS knowledge base
│   ├── query_kb.py                # Test RAG retrieval
│   ├── evaluate_retrieval.py      # Evaluate retrieval quality
│   └── download_documents.py      # Download official documents
│
├── tests/                          # Automated tests
│
├── vectorstore/
│   └── faiss_index/               # Committed FAISS index
│
└── docs/
    └── MEMBER1_RAG_KNOWLEDGE_BASE.md
```
RAG Knowledge Base
The project uses a FAISS vector index containing information retrieved from official NMAMIT documents.
The RAG pipeline follows:
```text
Documents
    ↓
Text Extraction
    ↓
Preprocessing
    ↓
Chunking
    ↓
Sentence Transformer Embeddings
    ↓
FAISS Vector Store
    ↓
Semantic Retrieval
    ↓
Relevant Context
    ↓
LLM Answer
```
The FAISS index is already included in `vectorstore/`, so the application can perform retrieval without rebuilding the knowledge base.
Test the Knowledge Base
```bash
python scripts/query_kb.py "What is the minimum attendance required?"
```
The RAG module can also be used programmatically:
```python
from src.rag import retrieve_context

result = retrieve_context("How long is Internship-II?")

result["found"]    # Whether relevant information was found
result["context"]  # Retrieved context
result["sources"]  # Source citations
```
Rebuilding the Knowledge Base
If the source documents are changed, the FAISS index can be rebuilt using:
```bash
python scripts/build_index.py
```
To perform a dry run and view chunking statistics:
```bash
python scripts/build_index.py --dry-run
```
Retrieval evaluation can be run using:
```bash
python scripts/evaluate_retrieval.py
```
Data Sources
The knowledge base is built from official public NMAMIT documents.
Document metadata and source URLs are maintained in:
```text
data/sources.json
```
The original documents are stored under:
```text
data/raw/pdfs/
```
The processed text and guideline content are stored under:
```text
data/raw/text/
```
Important Note
The application requires a Groq API key for LLM-powered features.
Each developer or user running the project locally should configure their own API key in `.env`.
The `.env` file is intentionally excluded from version control to protect API credentials.