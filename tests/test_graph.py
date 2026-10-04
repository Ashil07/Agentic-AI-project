"""Member 3 tests. Run from the project root:  python -m pytest tests/test_graph.py -v
No API key needed; the retriever is faked with the same shape as Member 1's RetrievedChunk."""
from dataclasses import dataclass

import pytest

import src.graph.agent_graph as ag
from src.graph.agent_graph import run_turn

PLAN_MSG = "Create a study plan for DBMS, OS, CN and TOC. I have 4 hours per day and my exam is in three weeks."


@dataclass
class FakeChunk:  # same fields the graph uses from RetrievedChunk
    text: str
    source: str
    citation: str


@pytest.fixture
def no_docs(monkeypatch):
    monkeypatch.setattr(ag, "_retrieve", lambda q, k=4: [])


@pytest.fixture
def with_docs(monkeypatch):
    chunk = FakeChunk("[hdr]\nStudents need 75% attendance to appear for exams.", "regs.pdf",
                      "NMAMIT Regulations - Attendance (p.10)")
    monkeypatch.setattr(ag, "_retrieve", lambda q, k=4: [chunk])
    monkeypatch.setattr(ag, "_answer_question", None)
    monkeypatch.setattr(ag, "LLM", None)


def test_unknown_question_is_not_invented(no_docs):
    r = run_turn("Who won the cricket world cup?")
    assert r["intent"] == "question" and r["answer"] == ag.NOT_FOUND


def test_question_returns_sources(with_docs):
    r = run_turn("What is the minimum attendance required?")
    assert r["intent"] == "question"
    assert r["sources"] == ["NMAMIT Regulations - Attendance (p.10)"]


def test_new_plan_and_modifications():
    r = run_turn(PLAN_MSG)
    assert r["intent"] == "plan_new"
    assert r["plan_spec"]["subjects"] == ["DBMS", "OS", "CN", "TOC"]
    assert r["plan_spec"]["hours_per_day"] == 4
    r = run_turn("I only have 2 hours on Wednesday.", plan_spec=r["plan_spec"])
    assert r["intent"] == "plan_modify" and r["plan_spec"]["weekday_hours"][2] == 2.0
    r = run_turn("Add Maths and remove CN", plan_spec=r["plan_spec"])
    assert "Maths" in r["plan_spec"]["subjects"] and "CN" not in r["plan_spec"]["subjects"]
    r = run_turn("No study on Sunday", plan_spec=r["plan_spec"])
    assert "rest day" in r["answer"].lower()


def test_missing_info_is_requested():
    assert "still need" in run_turn("Create a study plan for DBMS")["answer"]


def test_question_while_plan_exists_goes_to_rag(no_docs):
    spec = run_turn(PLAN_MSG)["plan_spec"]
    assert run_turn("What is the minimum attendance required?", plan_spec=spec)["intent"] == "question"
