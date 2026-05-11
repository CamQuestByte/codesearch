"""
Pre-compute query embeddings and cache them to disk.

The ef_sweep script (and any future queries-only experiment) reads from this
cache to skip the embedding step. The cache key includes the embedding model
name, so changing EMBEDDING_MODEL invalidates this cache automatically and
forces a recompute.

Usage:
    uv run python scripts/cache_query_vectors.py
    uv run python scripts/cache_query_vectors.py --recompute
"""

from __future__ import annotations

import argparse
import os
import pickle
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

import numpy as np
from sentence_transformers import SentenceTransformer

from codesearch.config import EMBEDDING_MODEL
from codesearch.data import load_codesearch

CACHE_DIR = ".cache"


def cache_path(model_name: str) -> str:
    safe = model_name.replace("/", "_")
    return os.path.join(CACHE_DIR, f"query_vectors_{safe}.pkl")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Cache query embeddings so the ef sweep can skip the encode step."
    )
    parser.add_argument(
        "--recompute",
        action="store_true",
        help="Re-embed even if the cache already exists.",
    )
    args = parser.parse_args()

    path = cache_path(EMBEDDING_MODEL)
    if os.path.exists(path) and not args.recompute:
        size_mb = os.path.getsize(path) / 1024 / 1024
        print(f"[skip] Cache already exists at {path} ({size_mb:.1f} MB).")
        print("       Pass --recompute to rebuild.")
        return

    os.makedirs(CACHE_DIR, exist_ok=True)

    print("[1/3] Loading eval queries (test split only)...")
    _, queries = load_codesearch(n=-1, queries_only=True)

    print(f"[2/3] Loading embedding model: {EMBEDDING_MODEL}")
    model = SentenceTransformer(EMBEDDING_MODEL)

    print(f"[3/3] Embedding {len(queries):,} queries...")
    t0 = time.perf_counter()
    vectors = model.encode(
        [q["query"] for q in queries],
        normalize_embeddings=True,
        show_progress_bar=True,
        batch_size=256,
    )
    elapsed = time.perf_counter() - t0
    print(f"  Done in {elapsed:.1f}s ({elapsed / len(queries) * 1000:.2f} ms/query)")

    print(f"Writing cache to {path}...")
    with open(path, "wb") as f:
        pickle.dump(
            {
                "model": EMBEDDING_MODEL,
                "queries": queries,
                "vectors": np.asarray(vectors, dtype=np.float32),
            },
            f,
            protocol=pickle.HIGHEST_PROTOCOL,
        )
    size_mb = os.path.getsize(path) / 1024 / 1024
    print(f"Cached {len(queries):,} query vectors ({size_mb:.1f} MB).")


if __name__ == "__main__":
    main()
