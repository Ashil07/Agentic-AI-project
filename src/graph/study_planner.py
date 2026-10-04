"""
study_planner.py  --  Member 3: Study Planner (no LLM required, LLM optional)

Design idea (important for the demo):
  * The plan is stored as a small, structured "PlanSpec" (subjects, hours/day,
    exam date, per-weekday overrides).
  * The timetable is ALWAYS generated from the spec by plain Python
    (build_plan). So plans are reliable and never "hallucinated".
  * A modification request ("I only have 2 hours on Wednesday") just edits the
    spec and regenerates the plan.
  * Understanding the student's sentence uses regex first; if something is
    missing and an LLM is supplied, the LLM fills the gaps (JSON output).
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field, asdict
from datetime import date, timedelta

WEEKDAYS = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
ALIASES = {"mon": 0, "tue": 1, "tues": 1, "wed": 2, "thu": 3, "thur": 3,
           "thurs": 3, "fri": 4, "sat": 5, "sun": 6}
NUM_WORDS = {"a": 1, "an": 1, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
             "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10}
MONTHS = {m: i + 1 for i, m in enumerate(
    ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"])}


# --------------------------------------------------------------------------
# Data model
# --------------------------------------------------------------------------
@dataclass
class PlanSpec:
    subjects: list[str]
    hours_per_day: float
    start_date: str                      # ISO yyyy-mm-dd
    exam_date: str                       # ISO yyyy-mm-dd (exam day itself is NOT a study day)
    weekday_hours: dict[int, float] = field(default_factory=dict)   # 0=Mon ... 6=Sun
    date_hours: dict[str, float] = field(default_factory=dict)      # specific dates
    block_hours: float = 1.0             # length of one study block

    def to_dict(self) -> dict:
        return asdict(self)

    @staticmethod
    def from_dict(d: dict) -> "PlanSpec":
        d = dict(d)
        d["weekday_hours"] = {int(k): float(v) for k, v in d.get("weekday_hours", {}).items()}
        d["date_hours"] = {str(k): float(v) for k, v in d.get("date_hours", {}).items()}
        return PlanSpec(**d)


# --------------------------------------------------------------------------
# Small parsing helpers
# --------------------------------------------------------------------------
def _num(token: str) -> int:
    token = token.lower()
    return NUM_WORDS[token] if token in NUM_WORDS else int(token)


def parse_hours(text: str) -> float | None:
    m = re.search(r"(\d+(?:\.\d+)?)\s*(?:hours?|hrs?|h)\b", text, re.I)
    return float(m.group(1)) if m else None


def weekdays_in(text: str) -> list[int]:
    t = text.lower()
    found: list[int] = []
    if re.search(r"\bweekends?\b", t):
        found += [5, 6]
    if re.search(r"\bweekdays?\b", t):
        found += [0, 1, 2, 3, 4]
    for m in re.finditer(r"\b(" + "|".join(WEEKDAYS + list(ALIASES)) + r")\b", t):
        w = m.group(1)
        found.append(WEEKDAYS.index(w) if w in WEEKDAYS else ALIASES[w])
    return sorted(set(found))


def parse_date_text(text: str, today: date) -> date | None:
    """Understands: 'in 3 weeks', 'tomorrow', 2026-11-20, 20/11/2026, '20 Nov', 'Nov 20 2026'."""
    t = text.lower()
    m = re.search(r"\bin\s+(\d+|a|an|one|two|three|four|five|six|seven|eight|nine|ten)\s+(day|week|month)s?\b", t)
    if m:
        n, unit = _num(m.group(1)), m.group(2)
        return today + timedelta(days=n * {"day": 1, "week": 7, "month": 30}[unit])
    if re.search(r"\bnext week\b", t):
        return today + timedelta(days=7)
    if re.search(r"\btomorrow\b", t):
        return today + timedelta(days=1)
    m = re.search(r"(\d{4})-(\d{2})-(\d{2})", t)
    if m:
        return date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
    m = re.search(r"\b(\d{1,2})[/-](\d{1,2})[/-](\d{4})\b", t)          # dd/mm/yyyy (India)
    if m:
        try:
            return date(int(m.group(3)), int(m.group(2)), int(m.group(1)))
        except ValueError:
            return None
    mon = "(" + "|".join(MONTHS) + r")[a-z]*"
    m = re.search(r"\b(\d{1,2})(?:st|nd|rd|th)?\s+" + mon + r"(?:\s+(\d{4}))?", t)
    d, mo, y = (int(m.group(1)), MONTHS[m.group(2)], m.group(3)) if m else (None, None, None)
    if not m:
        m = re.search(mon + r"\s+(\d{1,2})(?:st|nd|rd|th)?(?:\s+(\d{4}))?", t)
        if m:
            mo, d, y = MONTHS[m.group(1)], int(m.group(2)), m.group(3)
    if d and mo:
        year = int(y) if y else today.year
        try:
            res = date(year, mo, d)
            if not y and res < today:
                res = date(year + 1, mo, d)
            return res
        except ValueError:
            return None
    return None


def parse_subjects(text: str) -> list[str]:
    m = re.search(
        r"\bfor\s+(.+?)(?=[.;]|\bI\b|\bmy exams?\b|\bexams?\b|\bwith\b|,\s*\d|\s\d+\s*(?:hours?|hrs?)|$)",
        text, re.I | re.S)
    if not m:
        return []
    return clean_subject_list(m.group(1))


def clean_subject_list(chunk: str) -> list[str]:
    out = []
    for tok in re.split(r",|&|/|\band\b", chunk, flags=re.I):
        tok = re.sub(r"^\s*(my|the|all)\s+", "", tok.strip(), flags=re.I).strip(" .")
        if not tok or tok.lower() in {"exam", "exams", "me"} or len(tok.split()) > 5:
            continue
        if tok.islower() and len(tok) <= 4 and tok.lower() not in {"math", "java", "chem"}:
            tok = tok.upper()
        out.append(tok)
    return out


# --------------------------------------------------------------------------
# Optional LLM help (only used to fill gaps the regex could not understand)
# --------------------------------------------------------------------------
def _llm_json(llm, prompt: str) -> dict | None:
    if llm is None:
        return None
    try:
        raw = llm.invoke(prompt)
        raw = getattr(raw, "content", str(raw))
        raw = re.sub(r"```(?:json)?", "", raw).strip()
        return json.loads(raw[raw.index("{"): raw.rindex("}") + 1])
    except Exception:
        return None


def parse_request(text: str, today: date | None = None, llm=None) -> dict:
    """Returns dict with keys subjects, hours, exam (date|None). Missing = empty/None."""
    today = today or date.today()
    info = {"subjects": parse_subjects(text), "hours": parse_hours(text),
            "exam": parse_date_text(text, today)}
    if llm is not None and (not info["subjects"] or info["hours"] is None or info["exam"] is None):
        data = _llm_json(llm, (
            f"Today is {today.isoformat()}. Extract study-plan details from the message below. "
            'Reply with JSON only: {"subjects": [str], "hours_per_day": number|null, '
            '"exam_date": "YYYY-MM-DD"|null}. Use null if not stated.\n\nMessage: ' + text))
        if data:
            info["subjects"] = info["subjects"] or [str(s) for s in data.get("subjects") or []]
            if info["hours"] is None and data.get("hours_per_day"):
                info["hours"] = float(data["hours_per_day"])
            if info["exam"] is None and data.get("exam_date"):
                try:
                    info["exam"] = date.fromisoformat(data["exam_date"])
                except ValueError:
                    pass
    return info


def make_spec(text: str, today: date | None = None, llm=None) -> tuple[PlanSpec | None, list[str]]:
    """Returns (spec, missing_fields). spec is None while something is missing."""
    today = today or date.today()
    info = parse_request(text, today, llm)
    missing = []
    if not info["subjects"]:
        missing.append("which subjects you want to study")
    if info["hours"] is None:
        missing.append("how many hours per day you can study")
    if info["exam"] is None or info["exam"] <= today:
        missing.append("your exam date (e.g. 'in three weeks' or '2026-11-20')")
    if missing:
        return None, missing
    return PlanSpec(subjects=info["subjects"], hours_per_day=info["hours"],
                    start_date=today.isoformat(), exam_date=info["exam"].isoformat()), []


# --------------------------------------------------------------------------
# Plan generation  (deterministic)
# --------------------------------------------------------------------------
def hours_for(spec: PlanSpec, d: date) -> float:
    if d.isoformat() in spec.date_hours:
        return spec.date_hours[d.isoformat()]
    if d.weekday() in spec.weekday_hours:
        return spec.weekday_hours[d.weekday()]
    return spec.hours_per_day


def build_plan(spec: PlanSpec) -> list[dict]:
    """One dict per day: {date, weekday, hours, phase, sessions: {subject: hours}}."""
    start, exam = date.fromisoformat(spec.start_date), date.fromisoformat(spec.exam_date)
    days, d = [], start
    while d < exam:
        days.append(d)
        d += timedelta(days=1)
    if not days:
        raise ValueError("Exam date must be after the start date.")
    n, subjects = len(days), spec.subjects
    revision_days = 2 if n >= 7 else (1 if n >= 3 else 0)
    rot = 0                                    # rotation continues across days -> fair coverage
    plan = []
    for i, day in enumerate(days):
        hrs = hours_for(spec, day)
        if i >= n - revision_days:
            phase = "Revision"
        elif i >= int(n * 0.6):
            phase = "Practice & problem solving"
        else:
            phase = "Learn new topics"
        sessions: dict[str, float] = {}
        remaining = hrs
        while remaining > 1e-9:
            block = min(spec.block_hours, remaining)
            subj = subjects[rot % len(subjects)]
            sessions[subj] = round(sessions.get(subj, 0) + block, 2)
            remaining -= block
            rot += 1
        plan.append({"date": day.isoformat(), "weekday": WEEKDAYS[day.weekday()].title(),
                     "hours": hrs, "phase": phase, "sessions": sessions})
    return plan


def format_plan(spec: PlanSpec, plan: list[dict]) -> str:
    lines = [f"**Study plan** ({spec.start_date} → exam on {spec.exam_date})", "",
             "| Date | Day | Hours | Focus | Subjects |", "|---|---|---|---|---|"]
    totals: dict[str, float] = {s: 0 for s in spec.subjects}
    for p in plan:
        if p["sessions"]:
            what = ", ".join(f"{s} ({h:g}h)" for s, h in p["sessions"].items())
            for s, h in p["sessions"].items():
                totals[s] += h
        else:
            what = "Rest day"
        lines.append(f"| {p['date']} | {p['weekday'][:3]} | {p['hours']:g} | {p['phase']} | {what} |")
    lines += ["", "**Total hours per subject:** " + ", ".join(f"{s}: {h:g}h" for s, h in totals.items())]
    return "\n".join(lines)


def validate_plan(spec: PlanSpec, plan: list[dict]) -> list[str]:
    """Plan-review checks. Returns a list of warnings (empty list = plan is fine)."""
    warnings = []
    for p in plan:
        if sum(p["sessions"].values()) > p["hours"] + 1e-6:
            warnings.append(f"{p['date']}: planned more hours than available.")
    covered = {s for p in plan for s in p["sessions"]}
    for s in spec.subjects:
        if s not in covered:
            warnings.append(f"'{s}' is not covered at all - add more study time or days.")
    total = sum(p["hours"] for p in plan)
    if total < 2 * len(spec.subjects):
        warnings.append(f"Only {total:g} study hours in total for {len(spec.subjects)} subjects - this is very tight.")
    return warnings


# --------------------------------------------------------------------------
# Plan modification
# --------------------------------------------------------------------------
_OFF = r"\b(skip|no study|can'?t study|cannot study|off|rest|free|busy|holiday|no time)\b"


def apply_modification(spec: PlanSpec, text: str, today: date | None = None,
                       llm=None) -> tuple[PlanSpec, list[str]]:
    """Edit the spec from a natural-language request. Returns (spec, human-readable changes)."""
    today = today or date.today()
    changes: list[str] = []
    t = text.lower()
    days, hours = weekdays_in(t), parse_hours(t)

    if "exam" in t:
        nd = parse_date_text(t, today)
        if nd and nd > today:
            spec.exam_date = nd.isoformat()
            changes.append(f"Exam date set to {nd.isoformat()}.")
    elif days:
        if hours is not None:
            for wd in days:
                spec.weekday_hours[wd] = hours
            changes.append(f"{', '.join(WEEKDAYS[w].title() for w in days)}: {hours:g} hour(s).")
        elif re.search(_OFF, t):
            for wd in days:
                spec.weekday_hours[wd] = 0.0
            changes.append(f"{', '.join(WEEKDAYS[w].title() for w in days)}: rest day.")
    elif hours is not None and re.search(r"per day|a day|daily|each day|every day|/day", t):
        spec.hours_per_day = hours
        changes.append(f"Default study time set to {hours:g} hours/day.")
    elif hours is not None:
        d = parse_date_text(t, today)
        if d and today <= d < date.fromisoformat(spec.exam_date):
            spec.date_hours[d.isoformat()] = hours
            changes.append(f"{d.isoformat()}: {hours:g} hour(s).")

    m = re.search(r"\badd\s+(?!more\b)(.+?)(?:\s+(?:to|in|for)\b|[.;]|$)", text, re.I)
    if m:
        for s in clean_subject_list(m.group(1)):
            if s.lower() not in [x.lower() for x in spec.subjects]:
                spec.subjects.append(s)
                changes.append(f"Added subject {s}.")
    m = re.search(r"\b(?:remove|drop)\s+(.+?)(?:\s+from\b|[.;]|$)", text, re.I)
    if m:
        for s in clean_subject_list(m.group(1)):
            keep = [x for x in spec.subjects if x.lower() != s.lower()]
            if len(keep) < len(spec.subjects) and keep:
                spec.subjects = keep
                changes.append(f"Removed subject {s}.")

    if not changes and llm is not None:                       # LLM fallback for unusual phrasing
        data = _llm_json(llm, (
            f"Today is {today.isoformat()}. A student wants to change their study plan. Reply with JSON only: "
            '{"weekday_hours": {"monday": number, ...}, "hours_per_day": number|null, '
            '"add_subjects": [str], "remove_subjects": [str], "exam_date": "YYYY-MM-DD"|null}. '
            "Use {} / [] / null for things not mentioned.\n\nRequest: " + text))
        if data:
            for name, h in (data.get("weekday_hours") or {}).items():
                if name.lower() in WEEKDAYS:
                    spec.weekday_hours[WEEKDAYS.index(name.lower())] = float(h)
                    changes.append(f"{name.title()}: {float(h):g} hour(s).")
            if data.get("hours_per_day"):
                spec.hours_per_day = float(data["hours_per_day"])
                changes.append(f"Default study time set to {spec.hours_per_day:g} hours/day.")
            for s in data.get("add_subjects") or []:
                if s.lower() not in [x.lower() for x in spec.subjects]:
                    spec.subjects.append(s)
                    changes.append(f"Added subject {s}.")
            for s in data.get("remove_subjects") or []:
                keep = [x for x in spec.subjects if x.lower() != s.lower()]
                if keep and len(keep) < len(spec.subjects):
                    spec.subjects = keep
                    changes.append(f"Removed subject {s}.")
            if data.get("exam_date"):
                try:
                    nd = date.fromisoformat(data["exam_date"])
                    if nd > today:
                        spec.exam_date = nd.isoformat()
                        changes.append(f"Exam date set to {nd.isoformat()}.")
                except ValueError:
                    pass
    return spec, changes
