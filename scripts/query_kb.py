"""Ask the knowledge base a question from the command line (no LLM involved).

Usage (from the project root):
    python scripts/query_kb.py "What is the minimum attendance required?"
    python scripts/query_kb.py "syllabus of big data analytics" -k 3
    python scripts/query_kb.py              # interactive mode, type 'exit' to quit
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.rag import config, retrieve  # noqa: E402


def show(question: str, k: int, min_score: float | None) -> None:
    chunks = retrieve(question, k=k, min_score=min_score)
    print(f"\nQ: {question}")
    if not chunks:
        print(f"  -> No relevant chunks above threshold {config.RELEVANCE_THRESHOLD}. "
              "The assistant should answer: not found in NMAMIT documents.")
        return
    for i, c in enumerate(chunks, start=1):
        body = c.text.split("\n", 1)[-1].replace("\n", " ")
        print(f"\n  #{i}  score={c.score:.3f}  {c.citation}")
        print(f"      {body[:300]}{'...' if len(body) > 300 else ''}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("question", nargs="*", help="question to ask")
    parser.add_argument("-k", type=int, default=config.DEFAULT_TOP_K, help="number of chunks")
    parser.add_argument("--min-score", type=float, default=None, help="override relevance threshold")
    args = parser.parse_args()

    if args.question:
        show(" ".join(args.question), args.k, args.min_score)
        return
    print("NMAMIT knowledge base: type a question ('exit' to quit)")
    while True:
        try:
            q = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if q.lower() in {"exit", "quit", "q"}:
            break
        if q:
            show(q, args.k, args.min_score)


if __name__ == "__main__":
    main()
