"""
agent_graph.py  --  Member 3: LangGraph workflow

                      START
                        |
                 analyze_question
          ______________|_______________________
         |              |                      |
     (question)     (plan_new)           (plan_modify)
         |              |                      |
    retrieve        create_plan           modify_plan
         |              |___________ ___________|
    generate_answer                 |
         |                    review_plan
    review_answer                   |
         |                         END
        END

Public API for Member 4 (Streamlit):

    from src.graph import run_turn
    result = run_turn(message, history, plan_spec)
    result["answer"]     -> text to show in the chat
    result["plan_spec"]  -> store in st.session_state and pass back next turn
    result["intent"]     -> "question" | "plan_new" | "plan_modify"
    result["sources"]    -> list of source names (for questions)

Contracts with the other members (safe fallbacks are used if an import fails):

    Member 1:  src/rag      ->  retrieve(question, k=4) -> list[RetrievedChunk]   (already done)
    Member 2:  src/llm      ->  answer_question(question: str, chunks: list[str],
                                                history: list[dict]) -> str       (export it in src/llm/__init__.py)

Run from the PROJECT ROOT:   python -m src.graph.agent_graph
"""
from __future__ import annotations

import os
import re
from datetime import date
from typing import TypedDict

from langgraph.graph import StateGraph, START, END

try:
    from . import study_planner as sp          # normal use: python -m / import src.graph
except ImportError:                           # running the file directly
    import study_planner as sp

NOT_FOUND = ("I couldn't find this information in the available college documents. "
             "Please check with your department or the official college website.")

# ---------------------------------------------------------------- integrations
# Member 1 (already in the repo): src/rag/retriever.py -> retrieve(question, k) -> list[RetrievedChunk]
try:
    from src.rag import retrieve as _retrieve
except Exception as e:                       # missing packages / wrong working directory
    print("[agent_graph] RAG retriever not available:", e)
    _retrieve = None

# Member 2 (not in the repo yet): src/llm/ -> answer_question(question, context_chunks, history) -> str
try:
    from src.llm import answer_question as _answer_question
except Exception:
    _answer_question = None


def get_llm():
    """Groq LLM if configured, otherwise None (everything still works without it)."""
    if not os.getenv("GROQ_API_KEY"):
        return None
    try:
        from langchain_groq import ChatGroq
        return ChatGroq(model=os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile"), temperature=0)
    except Exception:
        return None


LLM = get_llm()


def _normalize_chunks(raw) -> list[dict]:
    """Accept RetrievedChunk (Member 1), LangChain Document, dict or plain str."""
    out = []
    for c in raw or []:
        if isinstance(c, str):
            out.append({"text": c, "source": "college documents"})
        elif isinstance(c, dict):
            out.append({"text": c.get("text") or c.get("page_content", ""),
                        "source": c.get("citation") or c.get("source", "college documents")})
        elif hasattr(c, "text"):                               # RetrievedChunk
            out.append({"text": c.text, "source": getattr(c, "citation", None) or c.source})
        else:                                                  # LangChain Document
            meta = getattr(c, "metadata", {}) or {}
            out.append({"text": c.page_content, "source": os.path.basename(str(meta.get("source", "college documents")))})
    return [c for c in out if c["text"].strip()]


# ---------------------------------------------------------------------- state
class State(TypedDict, total=False):
    message: str
    history: list[dict]          # [{"role": "user"|"assistant", "content": str}, ...]
    plan_spec: dict | None       # persisted by the UI between turns
    intent: str
    chunks: list[dict]
    answer: str
    sources: list[str]
    warnings: list[str]


# ---------------------------------------------------------------------- nodes
_WEEKDAY_RE = r"(?:monday|tuesday|wednesday|thursday|friday|saturday|sunday|mon|tue|tues|wed|thu|thur|thurs|fri|sat|sun|weekends?|weekdays?)"
PLAN_WORDS = re.compile(r"study plan|study schedule|study timetable|revision plan|revision schedule|timetable|"
                        r"plan (?:for|my)|schedule for", re.I)
CREATE_WORDS = re.compile(r"\b(create|make|generate|build|prepare|new|give me)\b", re.I)
MODIFY_HINT = re.compile(r"\b(only|skip|no study|instead|change|modify|update|add|remove|drop|postpone|"
                         r"reschedule|busy|free|can'?t|cannot|rest day|hours?)\b|\b" + _WEEKDAY_RE + r"\b", re.I)
QUESTION_START = re.compile(r"^\s*(what|when|where|who|which|why|how|is|are|does|do|tell me)\b", re.I)


def analyze_question(state: State) -> dict:
    """Decide: normal academic question, new study plan, or change to the existing plan."""
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
def retrieve_node(state: State) -> dict:
    chunks: list[dict] = []
    if _retrieve is not None:
        try:
            chunks = _normalize_chunks(_retrieve(state["message"]))
        except Exception as e:                                   # keep the demo alive
            print("[retrieve] error:", e)
    return {"chunks": chunks}


def generate_answer(state: State) -> dict:
    chunks = state.get("chunks", [])
    if not chunks:                                               # nothing retrieved -> never invent
        return {"answer": NOT_FOUND}
    texts = [c["text"] for c in chunks]
    history = state.get("history", [])
    if _answer_question is not None:                             # Member 2's chain
        return {"answer": _answer_question(state["message"], texts, history)}
    if LLM is not None:                                          # simple fallback
        ctx = "\n\n".join(texts)
        prompt = ("Answer ONLY from the context. If it is not there, say so.\n\n"
                  f"Context:\n{ctx}\n\nQuestion: {state['message']}")
        return {"answer": LLM.invoke(prompt).content}
    return {"answer": "(stub - no LLM configured) Relevant text found:\n\n" + texts[0][:500]}


def review_answer(state: State) -> dict:
    """Response review: block ungrounded answers, add sources."""
    chunks, answer = state.get("chunks", []), (state.get("answer") or "").strip()
    if not chunks or not answer:
        return {"answer": NOT_FOUND, "sources": []}
    if LLM is not None and answer != NOT_FOUND and os.getenv("REVIEW_WITH_LLM", "1") == "1":
        ctx = "\n\n".join(c["text"] for c in chunks)
        verdict = LLM.invoke(
            "You are a strict fact checker. Is every factual claim in the ANSWER supported by the CONTEXT? "
            f"Reply with exactly SUPPORTED or UNSUPPORTED.\n\nCONTEXT:\n{ctx}\n\nANSWER:\n{answer}").content
        if "UNSUPPORTED" in verdict.upper():
            return {"answer": NOT_FOUND, "sources": [],
                    "warnings": ["Answer rejected by review: not supported by documents."]}
    sources = sorted({c["source"] for c in chunks})
    return {"answer": answer, "sources": sources}


# ---- study-planner branch --------------------------------------------------
def create_plan(state: State) -> dict:
    spec, missing = sp.make_spec(state["message"], date.today(), LLM)
    if spec is None:
        return {"answer": "To create your study plan I still need: " + "; ".join(missing) + ".",
                "plan_spec": state.get("plan_spec")}
    plan = sp.build_plan(spec)
    return {"plan_spec": spec.to_dict(),
            "answer": sp.format_plan(spec, plan),
            "warnings": sp.validate_plan(spec, plan)}


def modify_plan(state: State) -> dict:
    spec = sp.PlanSpec.from_dict(state["plan_spec"])
    spec, changes = sp.apply_modification(spec, state["message"], date.today(), LLM)
    if not changes:
        return {"answer": ("I couldn't tell what to change. Try e.g. 'I only have 2 hours on Wednesday', "
                           "'no study on Sunday', 'add Maths', 'remove CN' or 'my exam is in 10 days'.")}
    plan = sp.build_plan(spec)
    return {"plan_spec": spec.to_dict(),
            "answer": "Updated your plan:\n- " + "\n- ".join(changes) + "\n\n" + sp.format_plan(spec, plan),
            "warnings": sp.validate_plan(spec, plan)}


def review_plan(state: State) -> dict:
    """Plan review: validation warnings (computed in the plan nodes) are shown to the student."""
    warnings = state.get("warnings") or []
    if warnings:
        return {"answer": state["answer"] + "\n\n**Review notes:**\n" + "\n".join(f"- {w}" for w in warnings)}
    return {}


# ---------------------------------------------------------------------- graph
def build_graph():
    g = StateGraph(State)
    g.add_node("analyze_question", analyze_question)
    g.add_node("retrieve", retrieve_node)
    g.add_node("generate_answer", generate_answer)
    g.add_node("review_answer", review_answer)
    g.add_node("create_plan", create_plan)
    g.add_node("modify_plan", modify_plan)
    g.add_node("review_plan", review_plan)

    g.add_edge(START, "analyze_question")
    g.add_conditional_edges("analyze_question", route_after_analysis,
                            {"question": "retrieve", "plan_new": "create_plan", "plan_modify": "modify_plan"})
    g.add_edge("retrieve", "generate_answer")
    g.add_edge("generate_answer", "review_answer")
    g.add_edge("review_answer", END)
    g.add_edge("create_plan", "review_plan")
    g.add_edge("modify_plan", "review_plan")
    g.add_edge("review_plan", END)
    return g.compile()


graph = build_graph()


def run_turn(message: str, history: list[dict] | None = None, plan_spec: dict | None = None) -> dict:
    """One conversation turn. Pass back result['plan_spec'] on the next call."""
    out = graph.invoke({"message": message, "history": history or [], "plan_spec": plan_spec})
    return {"answer": out.get("answer", ""), "intent": out.get("intent"),
            "plan_spec": out.get("plan_spec"), "sources": out.get("sources", []),
            "warnings": out.get("warnings", [])}


if __name__ == "__main__":                                       # quick CLI demo
    spec = None
    while True:
        msg = input("\nYou: ").strip()
        if msg.lower() in {"quit", "exit"}:
            break
        r = run_turn(msg, plan_spec=spec)
        spec = r["plan_spec"]
        print(f"\n[{r['intent']}]\n{r['answer']}")
