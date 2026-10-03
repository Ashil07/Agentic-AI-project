"""Download the official NMAMIT PDFs listed in data/sources.json.

The PDFs are already committed to the repo; this script exists so the
knowledge base can be re-created or refreshed from the official source
(nitte.edu.in). The college server can be slow, so the timeout is generous.

Usage (from the project root):
    python scripts/download_documents.py           # download only missing files
    python scripts/download_documents.py --force   # re-download everything
"""

import argparse
import json
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.rag import config  # noqa: E402


def download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(".part")
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (college-project-downloader)"})
    with urllib.request.urlopen(req, timeout=600) as resp, open(tmp, "wb") as f:
        while chunk := resp.read(1 << 16):
            f.write(chunk)
    if tmp.read_bytes()[:4] != b"%PDF":
        tmp.unlink()
        raise ValueError("downloaded file is not a PDF")
    tmp.replace(dest)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true", help="re-download files that already exist")
    args = parser.parse_args()

    urls = json.loads(config.SOURCES_FILE.read_text(encoding="utf-8"))["download_urls"]
    for rel_path, url in urls.items():
        dest = config.DATA_DIR / "raw" / rel_path
        if dest.exists() and not args.force:
            print(f"exists   {rel_path}")
            continue
        print(f"download {rel_path} ...", flush=True)
        try:
            download(url, dest)
            print(f"  ok ({dest.stat().st_size / 1e6:.1f} MB)")
        except Exception as exc:  # keep going if one file fails
            print(f"  FAILED: {exc}")


if __name__ == "__main__":
    main()
