"""
Build the BM25 index once and cache it to disk.

After this runs, any script that needs BM25 (compare_retrievers.py, future
M3 hybrid scripts, etc.) can do:

    bm25 = BM25Retriever.from_cache(".cache/bm25")

which loads in ~3-5 seconds instead of paying the ~2-3 min corpus build.

Usage:
    uv run python scripts/cache_bm25.py
    uv run python scripts/cache_bm25.py --recompute
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from codesearch.data import load_codesearch
from codesearch.retrievers.bm25_index import BM25Index

CACHE_DIR = ".cache/bm25"


def _dir_size_mb(path: str) -> float:
    total = 0
    for root, _, files in os.walk(path):
        for name in files:
            total += os.path.getsize(os.path.join(root, name))
    return total / 1024 / 1024


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build and cache the BM25 index over the full CodeSearchNet corpus."
    )
    parser.add_argument(
        "--recompute",
        action="store_true",
        help="Drop and rebuild the cache even if it already exists.",
    )
    args = parser.parse_args()

    if BM25Index.exists(CACHE_DIR) and not args.recompute:
        size = _dir_size_mb(CACHE_DIR)
        print(f"[skip] Cache already exists at {CACHE_DIR} ({size:.1f} MB).")
        print("       Pass --recompute to rebuild.")
        return

    if os.path.exists(CACHE_DIR) and args.recompute:
        print(f"[reset] Removing existing cache at {CACHE_DIR}...")
        shutil.rmtree(CACHE_DIR)

    print("[1/3] Loading full corpus from CodeSearchNet...")
    t0 = time.perf_counter()
    corpus, _ = load_codesearch(n=-1)
    print(f"  Corpus loaded in {time.perf_counter() - t0:.0f}s ({len(corpus):,} docs).")

    print("[2/3] Building BM25 index...")
    t0 = time.perf_counter()
    bm25_index = BM25Index.build(corpus)
    print(f"  Index built in {time.perf_counter() - t0:.0f}s.")

    print(f"[3/3] Saving to {CACHE_DIR}...")
    bm25_index.save(CACHE_DIR)
    size = _dir_size_mb(CACHE_DIR)
    print(f"Done. Cache size: {size:.1f} MB.")


if __name__ == "__main__":
    main()
