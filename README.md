# AI-Based College Academic Assistant (NMAMIT)

An AI-powered academic assistant for students of **NMAM Institute of Technology (NMAMIT), Nitte**.

The assistant answers academic questions using official college documents, maintains conversational context, performs mathematical calculations, and creates and modifies personalised study plans.

## Features

- **Academic Question Answering** using RAG over official NMAMIT documents
- **Conversational Memory** for follow-up questions
- **Calculator Tool** for mathematical calculations
- **Study Plan Generation** based on subjects, available study hours, and examination dates
- **Study Plan Modification** based on changes in the student's schedule
- **Source Citations** for retrieved college information
- **Hallucination Prevention** by refusing to answer college-specific questions when relevant information is not available in the knowledge base
- **Streamlit Web Interface**

## Tech Stack

- **Python**
- **Streamlit** – User interface
- **LangGraph** – Workflow and intent routing
- **LangChain** – LLM integration, prompts, tools and memory
- **Groq** – Large Language Model API
- **Sentence Transformers** – Text embeddings
- **FAISS** – Vector database for document retrieval
- **Pytest** – Automated testing

## System Architecture

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
                  /          \
                 /            \
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