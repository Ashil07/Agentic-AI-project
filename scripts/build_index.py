"""Build (or rebuild) the FAISS knowledge base from data/raw.

Usage (from the project root):
    python scripts/build_index.py
    python scripts/build_index.py --dry-run     # load + clean + chunk only, print stats

Run this again whenever you add or change a document in data/raw
(and list new documents in data/sources.json).
"""

import argparse
import logging
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.rag import config  # noqa: E402
from src.rag.chunker import split_documents  # noqa: E402
from src.rag.loader import load_all_documents  # noqa: E402
from src.rag.preprocess import preprocess_documents  # noqa: E402


def dry_run() -> None:
    raw = load_all_documents()
    cleaned = preprocess_documents(raw)
    chunks = split_documents(cleaned)
    lengths = sorted(len(c.page_content) for c in chunks)
    print(f"\nPages/sections: {len(raw)} | cleaned segments: {len(cleaned)} | chunks: {len(chunks)}")
    print(f"Chunk length (chars): min {lengths[0]}, median {lengths[len(lengths) // 2]}, max {lengths[-1]}")
    print("\nChunks per document:")
    for source, n in Counter(c.metadata["source"] for c in chunks).most_common():
        print(f"  {n:5d}  {source}")
    courses = {c.metadata["section"] for c in chunks if c.metadata.get("course_code")}
    print(f"\nDistinct courses detected: {len(courses)} (e.g. {sorted(courses)[:5]})")
    print("\nSample chunk:\n" + "-" * 60 + f"\n{chunks[len(chunks) // 3].page_content}\n" + "-" * 60)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dry-run", action="store_true", help="only load/clean/chunk and print statistics")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
    if args.dry_run:
        dry_run()
        return

    from src.rag.vector_store import build_vector_store

    build_vector_store()
    print(f"\nDone. Index saved in {config.VECTORSTORE_DIR}")
    print("Try it:  python scripts/query_kb.py \"What is the minimum attendance required?\"")


if __name__ == "__main__":
    main()
