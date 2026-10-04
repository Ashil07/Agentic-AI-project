import os
import re

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import ToolMessage
from langchain_core.tools import StructuredTool

from src.rag import retrieve_context
from .memory import get_memory
from .calculator import calculate


load_dotenv()


llm = ChatGroq(
    model="openai/gpt-oss-20b",
    api_key=os.getenv("GROQ_API_KEY"),
    temperature=0
)


calculator_tool = StructuredTool.from_function(
    func=calculate,
    name="calculator",
    description="Use this tool to calculate mathematical expressions."
)

llm_with_tools = llm.bind_tools([calculator_tool])


prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are an AI-based college academic assistant for NMAMIT.

Answer the student's question using the provided college context.

If the answer is not available in the provided context, clearly say:
"I could not find relevant information in the available NMAMIT documents."

Do not make up college-specific information.

Use the calculator tool when the student asks for a mathematical calculation.

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

    response = llm_with_tools.invoke(messages)

    if response.tool_calls:
        messages.append(response)

        for tool_call in response.tool_calls:
            result = calculator_tool.invoke(tool_call["args"])

            messages.append(
                ToolMessage(
                    content=str(result),
                    tool_call_id=tool_call["id"]
                )
            )

        response = llm_with_tools.invoke(messages)

    return response.content


def answer_question(question, session_id="default"):
    memory = get_memory(session_id)

    # Allow mathematical questions to use the calculator
    # even when no NMAMIT document is relevant.
    is_math = bool(re.fullmatch(r"[\d\s+\-*/().%]+", question.strip()) or 
                   re.search(r"\b\d+\s*[+\-*/]\s*\d+\b", question) or
                   re.search(r"\b(calculate|math|add|subtract|multiply|divide|sum|what is \d+)\b", question.lower()))
    
    if is_math:
        history = memory.messages

        answer = ask_llm(
            question=question,
            context="",
            history=history
        )

        memory.add_user_message(question)
        memory.add_ai_message(answer)

        return {
            "answer": answer,
            "sources": []
        }

    result = retrieve_context(question)

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

if __name__ == "__main__":
    question = input("Ask your question: ")
    result = answer_question(question)

    print("\nAnswer:")
    print(result["answer"])