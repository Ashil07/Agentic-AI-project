"""
Integration tests for the LangGraph workflow.
Run from the project root:  python -m pytest tests/test_graph.py -v
These tests require GROQ_API_KEY to be set in the .env file.
"""
import os
import pytest

from dotenv import load_dotenv

from src.graph.agent_graph import run_turn

load_dotenv()

# Skip tests if LLM is not configured, otherwise they will fail
pytestmark = pytest.mark.skipif(
    not os.getenv("GROQ_API_KEY"),
    reason="GROQ_API_KEY is not set. Real integration tests require an LLM."
)

PLAN_MSG = "Create a study plan for DBMS, OS, CN and TOC. I have 4 hours per day and my exam is in three weeks."

def test_academic_qa():
    """Test standard academic Q&A uses Member 1 RAG and Member 2 LLM."""
    # Assuming there's something about credits or attendance in the documents
    r = run_turn("What is the minimum attendance required?", session_id="test_qa")
    assert r["intent"] == "question"
    assert r["answer"] != ""
    assert isinstance(r["sources"], list)
    # The agent should return real sources instead of empty list for a valid document query
    assert len(r["sources"]) > 0
    # Make sure sources contain full metadata dictionaries
    assert isinstance(r["sources"][0], dict)
    assert "citation" in r["sources"][0]

def test_calculator_behavior():
    """Test that calculator queries route correctly and return an answer."""
    r = run_turn("What is 15 * 42?", session_id="test_calc")
    assert r["intent"] == "question"
    assert "630" in r["answer"]
    # Calculator queries don't need NMAMIT sources
    assert r["sources"] == []

def test_study_plan_creation():
    """Test that study plan creation generates a plan_spec."""
    r = run_turn(PLAN_MSG, session_id="test_plan")
    assert r["intent"] == "plan_new"
    
    spec = r["plan_spec"]
    assert spec is not None
    assert "DBMS" in spec["subjects"]
    assert "OS" in spec["subjects"]
    assert "CN" in spec["subjects"]
    assert "TOC" in spec["subjects"]
    assert spec["hours_per_day"] == 4.0
    
    # Check that answer contains formatted plan
    assert "DBMS" in r["answer"]

def test_study_plan_modification():
    """Test modifying an existing plan spec."""
    # Create plan first
    r1 = run_turn(PLAN_MSG, session_id="test_mod")
    spec = r1["plan_spec"]
    
    # Modify it
    r2 = run_turn("I only have 2 hours on Wednesday.", session_id="test_mod", plan_spec=spec)
    assert r2["intent"] == "plan_modify"
    new_spec = r2["plan_spec"]
    assert new_spec["weekday_hours"][2] == 2.0
    
    # Add/Remove subjects
    r3 = run_turn("Add Maths and remove CN", session_id="test_mod", plan_spec=new_spec)
    assert "Maths" in r3["plan_spec"]["subjects"]
    assert "CN" not in r3["plan_spec"]["subjects"]

def test_missing_info_is_requested():
    """Test that study plan prompts for missing info."""
    r = run_turn("Create a study plan for DBMS", session_id="test_missing")
    assert "still need" in r["answer"]
    assert r["plan_spec"] is None

def test_question_while_plan_exists_goes_to_qa():
    """Test that normal questions are routed to Q&A even if a plan exists."""
    r1 = run_turn(PLAN_MSG, session_id="test_routing")
    spec = r1["plan_spec"]
    
    r2 = run_turn("What is the minimum attendance required?", session_id="test_routing", plan_spec=spec)
    assert r2["intent"] == "question"
