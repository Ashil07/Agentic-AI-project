import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

llm = ChatGroq(
    model="llama-3.1-8b-instant",
    api_key=os.getenv("GROQ_API_KEY"),
    temperature=0
)

prompt = ChatPromptTemplate.from_template("""
You are an AI-based college academic assistant for NMAMIT.

Answer the student's question using the provided college context.

If the answer is not available in the provided context, clearly say:
"I could not find relevant information in the available NMAMIT documents."

Do not make up college-specific information.

College Context:
{context}

Student Question:
{question}
""")


def ask_llm(question, context=""):
    messages = prompt.format_messages(
        context=context,
        question=question
    )

    response = llm.invoke(messages)
    return response.content
