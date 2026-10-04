import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from src.rag import retrieve_context
from .memory import get_memory

load_dotenv()

llm = ChatGroq(
    model="llama-3.1-8b-instant",
    api_key=os.getenv("GROQ_API_KEY"),
    temperature=0
)

prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are an AI-based college academic assistant for NMAMIT.

Answer the student's question using the provided college context.

If the answer is not available in the provided context, clearly say:
"I could not find relevant information in the available NMAMIT documents."

Do not make up college-specific information.

College Context:
{context}
"""
    ),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{question}")
])


def ask_llm(question, context="", history=None):
    if history is None:
        history = []

    messages = prompt.format_messages(
        context=context,
        history=history,
        question=question
    )

    response = llm.invoke(messages)
    return response.content


def answer_question(question, session_id="default"):
    result = retrieve_context(question)
    memory = get_memory(session_id)

    if not result["found"]:
        answer = "I could not find relevant information in the available NMAMIT documents."
        memory.add_user_message(question)
        memory.add_ai_message(answer)

        return {
            "answer": answer,
            "sources": []
        }

    history = memory.messages

    answer = ask_llm(
        question=question,
        context=result["context"],
        history=history
    )

    memory.add_user_message(question)
    memory.add_ai_message(answer)

    return {
        "answer": answer,
        "sources": result["sources"]
    }
