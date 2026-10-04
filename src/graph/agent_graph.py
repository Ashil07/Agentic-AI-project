"""
agent_graph.py  --  Member 3: LangGraph workflow

                      START
                        |
                 analyze_question
          ______________|_______________________
         |              |                      |
     (question)     (plan_new)           (plan_modify)
         |              |                      |
    academic_qa     create_plan           modify_plan
         |              |___________ ___________|
         |                    |
         |               review_plan
         |                    |
         |____________________|
                    |
                   END

Public API for Member 4 (Streamlit):

    from src.graph import run_turn
    result = run_turn(message, session_id=session_id, plan_spec=plan_spec)
    result["answer"]     -> text to show in the chat
    result["plan_spec"]  -> store in st.session_state and pass back next turn
    result["intent"]     -> "question" | "plan_new" | "plan_modify"
    result["sources"]    -> list of source dicts (title, page, score, citation, url, source)
    result["warnings"]   -> list of plan validation warnings

Integration architecture:

    Streamlit UI
        |
        v
    LangGraph (this file)
        |--- Academic Question --> Member 2 answer_question(question, session_id)
        |                              |--- Member 1 RAG (retrieve_context)
        |                              |--- Member 2 LLM (openai/gpt-oss-20b via Groq)
        |                              |--- Member 2 Calculator
        |                              |--- Member 2 Memory (InMemoryChatMessageHistory)
        |
        |--- Study Plan -----------> Member 3 Study Planner (study_planner.py)

Run from the PROJECT ROOT:   python -m src.graph.agent_graph
"""
from __future__ import annotations

import os
import re
from datetime import date
from typing import TypedDict

from dotenv import load_dotenv
from langgraph.graph import StateGraph, START, END

try:
    from . import study_planner as sp
except ImportError:
    import study_planner as sp

# Load .env consistently so GROQ_API_KEY is available whether started
# via Streamlit, pytest, or direct python -m execution.
load_dotenv()

# ---------------------------------------------------------------------------
# Member 2 integration — single shared LLM and answer_question()
# ---------------------------------------------------------------------------
try:
    from src.llm import answer_question as _answer_question, llm as _llm
except Exception as e:
    print("[agent_graph] Member 2 (src.llm) not available:", e)
    _answer_question = None
    _llm = None

NOT_FOUND = ("I couldn't find this information in the available college documents. "
             "Please check with your department or the official college website.")


# ---------------------------------------------------------------------- state
class State(TypedDict, total=False):
    message: str
    session_id: str              # Memory keyed by session_id in Member 2
    plan_spec: dict | None       # persisted by the UI between turns
    intent: str
    answer: str
    sources: list[dict]          # full RAG source metadata dicts
    warnings: list[str]


# ---------------------------------------------------------------------- nodes
_WEEKDAY_RE = (r"(?:monday|tuesday|wednesday|thursday|friday|saturday|sunday"
               r"|mon|tue|tues|wed|thu|thur|thurs|fri|sat|sun|weekends?|weekdays?)")
PLAN_WORDS = re.compile(
    r"study plan|study schedule|study timetable|revision plan|revision schedule"
    r"|timetable|plan (?:for|my)|schedule for", re.I)
CREATE_WORDS = re.compile(r"\b(create|make|generate|build|prepare|new|give me)\b", re.I)
MODIFY_HINT = re.compile(
    r"\b(only|skip|no study|instead|change|modify|update|add|remove|drop|postpone"
    r"|reschedule|busy|free|can'?t|cannot|rest day|hours?)\b|\b" + _WEEKDAY_RE + r"\b", re.I)
QUESTION_START = re.compile(
    r"^\s*(what|when|where|who|which|why|how|is|are|does|do|tell me)\b", re.I)


def analyze_question(state: State) -> dict:
    """Decide: normal academic question, new study plan, or modify existing plan."""
    msg = state["message"]
    has_plan = bool(state.get("plan_spec"))

    if PLAN_WORDS.search(msg) and CREATE_WORDS.search(msg):
        intent = "plan_new"
    elif has_plan and MODIFY_HINT.search(msg) and not QUESTION_START.match(msg):
        intent = "plan_modify"
    elif PLAN_WORDS.search(msg):
        intent = "plan_new"
    else:
        intent = "question"

    return {"intent": intent}


def route_after_analysis(state: State) -> str:
    return state["intent"]


# ---- question branch -------------------------------------------------------
# Delegates entirely to Member 2.  Member 2 internally handles:
#   - RAG retrieval (Member 1)
#   - LLM generation
#   - Calculator tool calls
#   - Conversation memory (keyed by session_id)
#   - Source metadata return
# LangGraph does NOT duplicate any of this.

def academic_qa(state: State) -> dict:
    """Route academic questions and calculations to Member 2."""
    if _answer_question is None:
        raise RuntimeError(
            "Member 2 integration missing: src.llm.answer_question could not be imported. "
            "Ensure src/llm/groq_llm.py is present, GROQ_API_KEY is set, and "
            "dependencies are installed.")

    result = _answer_question(
        state["message"],
        session_id=state.get("session_id", "default"),
    )
    # Preserve full source metadata dicts (title, page, score, citation, url, ...)
    return {"answer": result["answer"], "sources": result["sources"]}


# ---- study-planner branch --------------------------------------------------
# The study planner uses regex first and the shared LLM only as a gap-filler.
# It works without an LLM for well-formed requests.

def create_plan(state: State) -> dict:
    """Create a new study plan from the student's message."""
    spec, missing = sp.make_spec(state["message"], date.today(), _llm)
    if spec is None:
        return {
            "answer": "To create your study plan I still need: "
                      + "; ".join(missing) + ".",
            "plan_spec": state.get("plan_spec"),
        }

    plan = sp.build_plan(spec)
    return {
        "plan_spec": spec.to_dict(),
        "answer": sp.format_plan(spec, plan),
        "warnings": sp.validate_plan(spec, plan),
    }


def modify_plan(state: State) -> dict:
    """Modify an existing study plan based on the student's request."""
    spec = sp.PlanSpec.from_dict(state["plan_spec"])
    spec, changes = sp.apply_modification(
        spec, state["message"], date.today(), _llm)

    if not changes:
        return {
            "answer": (
                "I couldn't tell what to change. Try e.g. "
                "'I only have 2 hours on Wednesday', "
                "'no study on Sunday', 'add Maths', 'remove CN' "
                "or 'my exam is in 10 days'."),
        }

    plan = sp.build_plan(spec)
    return {
        "plan_spec": spec.to_dict(),
        "answer": ("Updated your plan:\n- "
                   + "\n- ".join(changes)
                   + "\n\n" + sp.format_plan(spec, plan)),
        "warnings": sp.validate_plan(spec, plan),
    }


def review_plan(state: State) -> dict:
    """Append validation warnings to the plan answer."""
    warnings = state.get("warnings") or []
    if warnings:
        return {
            "answer": (state["answer"]
                       + "\n\n**Review notes:**\n"
                       + "\n".join(f"- {w}" for w in warnings)),
        }
    return {}


# ---------------------------------------------------------------------- graph
def build_graph():
    g = StateGraph(State)
    g.add_node("analyze_question", analyze_question)
    g.add_node("academic_qa", academic_qa)
    g.add_node("create_plan", create_plan)
    g.add_node("modify_plan", modify_plan)
    g.add_node("review_plan", review_plan)

    g.add_edge(START, "analyze_question")
    g.add_conditional_edges(
        "analyze_question", route_after_analysis,
        {"question": "academic_qa",
         "plan_new": "create_plan",
         "plan_modify": "modify_plan"})

    g.add_edge("academic_qa", END)
    g.add_edge("create_plan", "review_plan")
    g.add_edge("modify_plan", "review_plan")
    g.add_edge("review_plan", END)
    return g.compile()


graph = build_graph()


def run_turn(
    message: str,
    history: list | None = None,
    plan_spec: dict | None = None,
    session_id: str = "default",
) -> dict:
    """One conversation turn — the single entry point for Streamlit.

    Args:
        message:    The student's message.
        history:    Unused (kept for backward compatibility). Memory is
                    managed by Member 2 via session_id.
        plan_spec:  The current study-plan spec (pass back from previous turn).
        session_id: Session identifier for Member 2 conversation memory.

    Returns:
        dict with keys: answer, intent, plan_spec, sources, warnings.
    """
    out = graph.invoke({
        "message": message,
        "session_id": session_id,
        "plan_spec": plan_spec,
    })
    return {
        "answer": out.get("answer", ""),
        "intent": out.get("intent"),
        "plan_spec": out.get("plan_spec"),
        "sources": out.get("sources", []),
        "warnings": out.get("warnings", []),
    }


if __name__ == "__main__":                                       # quick CLI demo
    spec = None
    sid = "cli_user"
    while True:
        msg = input("\nYou: ").strip()
        if msg.lower() in {"quit", "exit"}:
            break
        r = run_turn(msg, session_id=sid, plan_spec=spec)
        spec = r["plan_spec"]
        print(f"\n[{r['intent']}]\n{r['answer']}")
