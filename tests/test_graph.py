"""
Integration tests for the LangGraph workflow (Member 3).
Run from the project root:  python -m pytest tests/test_graph.py -v

Tests are split into two groups:
  - Study planner tests: work offline (no API key needed).
  - Academic Q&A / calculator tests: require GROQ_API_KEY in .env.
"""
import os

import pytest
from dotenv import load_dotenv

load_dotenv()

from src.graph.agent_graph import run_turn

PLAN_MSG = (
    "Create a study plan for DBMS, OS and CN. "
    "I can study 3 hours per day and my exam is in three weeks."
)

needs_api_key = pytest.mark.skipif(
    not os.getenv("GROQ_API_KEY"),
    reason="GROQ_API_KEY is not set — skipping live LLM tests.",
)


# =========================================================================
# A. Academic question  (requires API key)
# =========================================================================
@needs_api_key
def test_academic_qa():
    """run_turn routes an academic question to Member 2 and returns sources."""
    r = run_turn("What are the attendance requirements?", session_id="t_qa")
    assert r["intent"] == "question"
    assert r["answer"] != ""
    assert isinstance(r["sources"], list)
    # RAG should find relevant NMAMIT documents
    if r["sources"]:
        # Full metadata dicts must be preserved
        assert isinstance(r["sources"][0], dict)
        assert "citation" in r["sources"][0]


# =========================================================================
# B. Unknown question  (requires API key)
# =========================================================================
@needs_api_key
def test_unknown_question_not_invented():
    """Questions outside NMAMIT documents should not produce hallucinated answers."""
    r = run_turn("Who won the FIFA World Cup in 2022?", session_id="t_unk")
    assert r["intent"] == "question"
    # Should indicate info not found rather than making something up
    answer_lower = r["answer"].lower()
    assert (
        "could not find" in answer_lower
        or "couldn't find" in answer_lower
        or "not available" in answer_lower
        or "no relevant" in answer_lower
        # Member 2 may phrase it differently but shouldn't name a winner
    ), f"Expected a not-found response, got: {r['answer'][:200]}"


# =========================================================================
# C. Calculator  (requires API key)
# =========================================================================
@needs_api_key
def test_calculator_pure_expression():
    """Pure math expression: '15 * 42' should return 630."""
    r = run_turn("15 * 42", session_id="t_calc1")
    assert r["intent"] == "question"
    assert "630" in r["answer"]
    assert r["sources"] == []


@needs_api_key
def test_calculator_natural_language():
    """Natural-language math: 'What is 15 * 42?' should also return 630."""
    r = run_turn("What is 15 * 42?", session_id="t_calc2")
    assert r["intent"] == "question"
    assert "630" in r["answer"]
    assert r["sources"] == []


# =========================================================================
# D. New study plan  (works offline — study planner uses regex)
# =========================================================================
def test_new_study_plan():
    """Creating a study plan returns a plan_spec and formatted plan."""
    r = run_turn(PLAN_MSG, session_id="t_plan")
    assert r["intent"] == "plan_new"

    spec = r["plan_spec"]
    assert spec is not None
    assert "DBMS" in spec["subjects"]
    assert "OS" in spec["subjects"]
    assert "CN" in spec["subjects"]
    assert spec["hours_per_day"] == 3.0

    # The answer should contain the formatted timetable
    assert "DBMS" in r["answer"]


# =========================================================================
# E. Plan modification  (works offline)
# =========================================================================
def test_plan_modification():
    """Modifying an existing plan updates the plan_spec correctly."""
    # Step 1: create
    r1 = run_turn(PLAN_MSG, session_id="t_mod")
    spec = r1["plan_spec"]
    assert spec is not None

    # Step 2: change hours on Wednesday
    r2 = run_turn(
        "I only have 2 hours on Wednesday.",
        session_id="t_mod", plan_spec=spec,
    )
    assert r2["intent"] == "plan_modify"
    assert r2["plan_spec"]["weekday_hours"][2] == 2.0

    # Step 3: add/remove subjects
    r3 = run_turn(
        "Add Maths and remove CN",
        session_id="t_mod", plan_spec=r2["plan_spec"],
    )
    assert "Maths" in r3["plan_spec"]["subjects"]
    assert "CN" not in r3["plan_spec"]["subjects"]

    # Step 4: rest day
    r4 = run_turn(
        "No study on Sunday",
        session_id="t_mod", plan_spec=r3["plan_spec"],
    )
    assert "rest day" in r4["answer"].lower()


# =========================================================================
# F. Missing info is requested  (works offline)
# =========================================================================
def test_missing_info_is_requested():
    """When the student doesn't give enough info, the planner asks for more."""
    r = run_turn("Create a study plan for DBMS", session_id="t_miss")
    assert "still need" in r["answer"]
    # plan_spec should remain None until enough info is provided
    assert r["plan_spec"] is None


# =========================================================================
# G. Question while plan exists routes to Q&A  (works offline as routing check)
# =========================================================================
def test_question_while_plan_exists_routes_to_qa():
    """Normal questions are routed to academic_qa even if a plan exists."""
    # Test intent classification only (does not call Member 2)
    from src.graph.agent_graph import analyze_question

    r1 = run_turn(PLAN_MSG, session_id="t_route")
    spec = r1["plan_spec"]

    result = analyze_question({
        "message": "What is the minimum attendance required?",
        "plan_spec": spec,
    })
    assert result["intent"] == "question"


# =========================================================================
# H. run_turn returns all expected keys
# =========================================================================
def test_run_turn_return_keys():
    """run_turn always returns the documented keys."""
    r = run_turn("Create a study plan for DBMS", session_id="t_keys")
    for key in ("answer", "intent", "plan_spec", "sources", "warnings"):
        assert key in r, f"Missing key: {key}"
