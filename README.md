# AI-Based College Academic Assistant (NMAMIT)

An AI-powered academic assistant for students of **NMAM Institute of
Technology (NMAMIT), Nitte**. The assistant answers academic questions
using official college documents, maintains conversational context,
performs mathematical calculations, and creates and modifies
personalised study plans.

## Features

-   **Academic Question Answering** using RAG over official NMAMIT
    documents
-   **Conversational Memory** for follow-up questions
-   **Calculator Tool** for mathematical calculations
-   **Study Plan Generation** based on subjects, available study hours,
    and examination dates
-   **Study Plan Modification** based on changes in the student's
    schedule
-   **Source Citations** for retrieved college information
-   **Hallucination Prevention** by refusing to answer college-specific
    questions when relevant information is not available in the
    knowledge base
-   **Streamlit Web Interface** for interacting with the assistant

## Tech Stack

-   **Python** -- Core programming language
-   **Streamlit** -- User interface
-   **LangGraph** -- Workflow orchestration and intent routing
-   **LangChain** -- LLM integration, prompts, tools, and memory
-   **Groq** -- Large Language Model API
-   **Sentence Transformers** -- Text embeddings
-   **FAISS** -- Vector database for document retrieval
-   **Pytest** -- Automated testing

## System Architecture

``` text
STUDENT
   |
   v
STREAMLIT UI
   |
   v
LANGGRAPH WORKFLOW
   |
   v
INTENT ANALYSIS
   /        \
  /          \
 v            v
ACADEMIC    STUDY PLAN
QUESTION
   |            |
   v            v
LANGCHAIN    STUDY PLANNER
   |
   +------------------+
   |                  |
   v                  v
RAG / KNOWLEDGE    CALCULATOR
BASE               TOOL
   |
   v
GROQ LLM
   |
   v
FINAL ANSWER + SOURCES
```

## Team Member Responsibilities

  -----------------------------------------------------------------------
  Member            Responsibility    Folder / File     Status
  ----------------- ----------------- ----------------- -----------------
  1                 RAG / Knowledge   `src/rag/`        Done
                    Base: documents →                   
                    chunks →                            
                    embeddings →                        
                    FAISS → retriever                   

  2                 LangChain / Groq  `src/llm/`        Done
                    LLM / prompts /                     
                    memory /                            
                    calculator tool                     

  3                 LangGraph         `src/graph/`      Done
                    workflow + study                    
                    planner                             

  4                 Streamlit UI,     `app.py`          Done
                    integration and                     
                    testing                             
  -----------------------------------------------------------------------

## Project Structure

``` text
Agentic-AI-project/
│
├── app.py
├── requirements.txt
├── README.md
├── .env.example
│
├── data/
│   └── ...
│
├── docs/
│   └── ...
│
├── scripts/
│   └── ...
│
├── src/
│   ├── graph/
│   │   ├── agent_graph.py
│   │   └── study_planner.py
│   │
│   ├── llm/
│   │   ├── groq_llm.py
│   │   ├── memory.py
│   │   └── calculator.py
│   │
│   └── rag/
│       └── ...
│
├── tests/
│   ├── test_graph.py
│   └── test_rag.py
│
└── vectorstore/
    └── ...
```

## Getting Started

### 1. Clone the repository

``` bash
git clone https://github.com/Ashil07/Agentic-AI-project.git
cd Agentic-AI-project
```

### 2. Create a virtual environment

``` bash
python -m venv .venv
```

Activate it on Windows PowerShell:

``` powershell
.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

``` bash
python -m pip install -r requirements.txt
```

### 4. Configure the API key

Create a `.env` file in the project root and add your Groq API key:

``` env
GROQ_API_KEY=your_groq_api_key_here
```

> **Important:** Never commit or push your `.env` file or API key to
> GitHub. The repository provides `.env.example` as a template.

### 5. Run the application

``` bash
streamlit run app.py
```

The application will open in your browser.

## Testing

Run the complete test suite with:

``` bash
python -m pytest
```

The project includes tests for:

-   Academic question answering
-   Unknown/out-of-context questions
-   Calculator functionality
-   Study plan generation
-   Study plan modification
-   Intent routing
-   RAG retrieval

## Example Capabilities

### Academic Questions

Ask questions based on the available NMAMIT knowledge base.

Example:

``` text
What is NMAMIT?
```

The assistant retrieves relevant information and provides the answer
along with source citations.

### Conversational Follow-ups

The assistant maintains conversation history, allowing follow-up
questions such as:

``` text
What courses are offered there?
```

### Calculations

The assistant can perform mathematical calculations using its calculator
tool.

Example:

``` text
What is 125 × 8?
```

### Study Planning

Students can request a personalised study plan by providing subjects,
available study hours, and examination dates.

Example:

``` text
Create a study plan for DBMS and Computer Networks.
I can study 3 hours per day and my exams are in 3 weeks.
```

The generated plan can later be modified when the student's availability
changes.

Example:

``` text
I can study only 2 hours on weekdays.
```

## Hallucination Prevention

For college-specific questions, the assistant relies on the available
NMAMIT knowledge base. If relevant information cannot be retrieved, it
does not invent an answer and instead responds:

> I could not find relevant information in the available NMAMIT
> documents.

This helps keep college-specific answers grounded in the provided
documents.

## Team

This project was developed as a team project using an agentic AI
architecture combining RAG, LangChain, LangGraph, an LLM, and a
Streamlit interface.
