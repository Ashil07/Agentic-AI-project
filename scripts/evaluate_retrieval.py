"""Measure retrieval quality and calibrate the relevance threshold.

Metrics:
  - Hit@k        : % of in-domain questions where an expected phrase appears in the top-k chunks
  - MRR          : mean reciprocal rank of the first chunk containing an expected phrase
  - OOD rejected : % of out-of-domain questions for which retrieve() returns nothing
  - Score ranges : top-1 cosine similarity for each group, used to choose RELEVANCE_THRESHOLD

Usage (from the project root):
    python scripts/evaluate_retrieval.py
    python scripts/evaluate_retrieval.py -k 5 --verbose
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.rag import config, retrieve  # noqa: E402

EVAL_FILE = config.PROJECT_ROOT / "tests" / "eval_questions.json"


def first_hit_rank(chunks, expect_any: list[str]) -> int | None:
    for rank, c in enumerate(chunks, start=1):
        text = c.text.lower()
        if any(phrase.lower() in text for phrase in expect_any):
            return rank
    return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("-k", type=int, default=config.DEFAULT_TOP_K)
    parser.add_argument("--verbose", action="store_true", help="print every question")
    args = parser.parse_args()

    data = json.loads(EVAL_FILE.read_text(encoding="utf-8"))

    hits, rr, in_scores, misses, in_rejected = 0, 0.0, [], [], 0
    for item in data["in_domain"]:
        chunks = retrieve(item["q"], k=args.k, min_score=0)
        if not retrieve(item["q"], k=args.k):  # uses the configured threshold
            in_rejected += 1
        rank = first_hit_rank(chunks, item["expect_any"])
        in_scores.append(chunks[0].score)
        if rank:
            hits += 1
            rr += 1 / rank
        else:
            misses.append(item["q"])
        if args.verbose:
            print(f"{'HIT ' if rank else 'MISS'} rank={rank} top1={chunks[0].score:.3f}  {item['q']}\n      -> {chunks[0].citation}")

    ood_scores, rejected = [], 0
    for q in data["out_of_domain"]:
        top = retrieve(q, k=1, min_score=0)[0].score
        ood_scores.append(top)
        if not retrieve(q, k=args.k):  # uses the configured threshold
            rejected += 1
        if args.verbose:
            print(f"OOD  top1={top:.3f}  {q}")

    college_scores = [retrieve(q, k=1, min_score=0)[0].score for q in data["college_but_not_in_documents"]]

    n_in, n_ood = len(data["in_domain"]), len(data["out_of_domain"])
    print("\n=========== Retrieval evaluation ===========")
    print(f"Embedding model     : {config.EMBEDDING_MODEL_NAME}")
    print(f"Chunk size/overlap  : {config.CHUNK_SIZE}/{config.CHUNK_OVERLAP}")
    print(f"In-domain questions : {n_in}")
    print(f"Hit@{args.k}               : {hits / n_in:.1%}")
    print(f"MRR                 : {rr / n_in:.3f}")
    print(f"Threshold           : {config.RELEVANCE_THRESHOLD}")
    print(f"OOD rejected        : {rejected}/{n_ood} ({rejected / n_ood:.0%})")
    print(f"In-domain wrongly rejected by threshold: {in_rejected}/{n_in}")
    print("\nTop-1 cosine similarity ranges:")
    print(f"  in-domain                 : min {min(in_scores):.3f}  avg {sum(in_scores) / n_in:.3f}  max {max(in_scores):.3f}")
    print(f"  out-of-domain             : min {min(ood_scores):.3f}  avg {sum(ood_scores) / n_ood:.3f}  max {max(ood_scores):.3f}")
    print(f"  college, not in documents : min {min(college_scores):.3f}  max {max(college_scores):.3f}"
          "  (the LLM prompt must handle these: score alone cannot)")
    gap_mid = (min(in_scores) + max(ood_scores)) / 2
    print(f"\nSuggested threshold (midpoint of the gap): {gap_mid:.2f}")
    if misses:
        print("\nMissed questions:")
        for q in misses:
            print(f"  - {q}")


if __name__ == "__main__":
    main()
