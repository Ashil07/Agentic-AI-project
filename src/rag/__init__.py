"""RAG / Knowledge Base module (Member 1).

Public API for the rest of the team:

    from src.rag import retrieve, retrieve_context, format_context, get_retriever

- retrieve(question, k=4)          -> list[RetrievedChunk]  (empty list = nothing relevant)
- retrieve_context(question, k=4)  -> {"found", "context", "sources", "question"}
- format_context(chunks)           -> str to put into the LLM prompt
- get_retriever(k=4)               -> LangChain VectorStoreRetriever for LCEL chains
- build_vector_store()             -> rebuild the FAISS index from data/raw
"""

from .retriever import (
    RetrievedChunk,
    format_context,
    get_retriever,
    has_relevant_context,
    retrieve,
    retrieve_context,
)
from .vector_store import build_vector_store, load_vector_store

__all__ = [
    "RetrievedChunk",
    "build_vector_store",
    "format_context",
    "get_retriever",
    "has_relevant_context",
    "load_vector_store",
    "retrieve",
    "retrieve_context",
]
