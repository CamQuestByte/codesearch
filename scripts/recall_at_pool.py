"""
Measure Recall@K of candidate pools — answers "what's the reranker's ceiling?"

For each query, the cross-encoder can only re-rank documents that the upstream
retriever put in the pool. So Recall@K of the pool is the upper bound on what
any reranker can achieve at top-K. This script reports that ceiling for four
pool strategies, at three pool sizes:

  - BM25 top-K
  - Dense top-K
  - RRF(BM25 top-K, Dense top-K) top-K       (k=60, missing rank = K+1)
  - Set union (BM25 top-K ∪ Dense top-K)     — absolute ceiling for any fusion

The gap between RRF and set-union tells how much ranking quality matters above
pool composition. The gap between RRF and the better of (BM25, Dense) tells
whether fusion is pulling its weight at that K.

Usage:
    uv run python scripts/cache_query_vectors.py   # one-time, if not done
    uv run python scripts/recall_at_pool.py
    uv run python scripts/recall_at_pool.py --max-queries -1   # full corpus
"""

from __future__ import annotations

import argparse
import os
import pickle
import random
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

import httpx
from tqdm import tqdm

from codesearch.config import (
    EMBEDDING_MODEL,
    QDRANT_API_KEY,
    QDRANT_COLLECTION,
    QDRANT_URL,
)
from codesearch.data import load_codesearch
from codesearch.retrievers.bm25 import BM25Retriever
from codesearch.retrievers.bm25_index import BM25Index
from codesearch.retrievers.hybrid import rrf_fuse

_SAMPLE_SEED = 42
_MAX_K = 100
_POOL_SIZES = (20, 50, 100)
_SEARCH_BATCH = 50
CACHE_DIR = ".cache"
BM25_CACHE_DIR = ".cache/bm25"


def cache_path(model_name: str) -> str:
    safe = model_name.replace("/", "_")
    return os.path.join(CACHE_DIR, f"query_vectors_{safe}.pkl")


def load_cached() -> tuple[list[dict], list[list[float]]]:
    path = cache_path(EMBEDDING_MODEL)
    if not os.path.exists(path):
        sys.exit(
            f"[error] Cache not found at {path}.\n"
            f"        Run: uv run python scripts/cache_query_vectors.py"
        )
    with open(path, "rb") as f:
        data = pickle.load(f)
    if data["model"] != EMBEDDING_MODEL:
        sys.exit(
            f"[error] Cache model mismatch. Re-run scripts/cache_query_vectors.py --recompute"
        )
    return data["queries"], data["vectors"].tolist()


def dense_search_batch(http: httpx.Client, vectors: list[list[float]], ef: int = 128) -> list[list[str]]:
    hits_all: list[list[str]] = []
    n_batches = (len(vectors) + _SEARCH_BATCH - 1) // _SEARCH_BATCH
    for i in tqdm(range(0, len(vectors), _SEARCH_BATCH), total=n_batches, desc="Dense search"):
        chunk = vectors[i : i + _SEARCH_BATCH]
        payload = {
            "searches": [
                {
                    "query": qv,
                    "limit": _MAX_K,
                    "params": {"hnsw_ef": ef},
                    "with_payload": ["doc_id"],
                }
                for qv in chunk
            ]
        }
        r = http.post(
            f"/collections/{QDRANT_COLLECTION}/points/query/batch",
            json=payload,
            timeout=60.0,
        )
        r.raise_for_status()
        for resp in r.json()["result"]:
            hits_all.append([p["payload"]["doc_id"] for p in resp["points"]])
    return hits_all


def _id_dicts(ids: list[str]) -> list[dict]:
    """Wrap doc-ids as minimal hit-dicts for rrf_fuse (which takes dicts)."""
    return [{"id": i} for i in ids]


def recall(hit_lists: list[list[str]], relevant: list[str], k: int) -> float:
    """Fraction of queries whose relevant doc appears in the top-k of its hit list."""
    n = len(hit_lists)
    if n == 0:
        return 0.0
    return sum(1 for hits, rel in zip(hit_lists, relevant) if rel in hits[:k]) / n


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Recall@K of candidate pools (BM25, Dense, RRF, set-union ceiling)."
    )
    parser.add_argument(
        "--max-queries",
        type=int,
        default=2000,
        help="Sample size (default: 2000; -1 = full eval set).",
    )
    args = parser.parse_args()

    # [1/4] Cached queries + vectors
    print("[1/4] Loading cached query vectors...")
    queries, vectors = load_cached()
    print(f"  Loaded {len(queries):,} queries.")

    if args.max_queries and args.max_queries > 0 and len(queries) > args.max_queries:
        random.seed(_SAMPLE_SEED)
        idx = random.sample(range(len(queries)), args.max_queries)
        queries = [queries[i] for i in idx]
        vectors = [vectors[i] for i in idx]
        print(f"  Sampled {len(queries):,} queries (seed={_SAMPLE_SEED}).")

    relevant = [q["relevant_id"] for q in queries]

    # [2/4] BM25 — from cache if available, else build from scratch
    if BM25Index.exists(BM25_CACHE_DIR):
        print(f"[2/4] Loading cached BM25 index from {BM25_CACHE_DIR}...")
        bm25 = BM25Retriever.from_cache(BM25_CACHE_DIR)
        print(f"  Loaded BM25 over {len(bm25.corpus):,} docs.")
    else:
        print("[2/4] No BM25 cache found — building from scratch (~2-3 min).")
        print(f"      Tip: run scripts/cache_bm25.py to skip this next time.")
        corpus, _ = load_codesearch(n=-1)
        bm25 = BM25Retriever(corpus)

    # [3/4] BM25 search
    print(f"[3/4] Running BM25 on {len(queries):,} queries (top-{_MAX_K})...")
    bm25_results = bm25.retrieve_batch([q["query"] for q in queries], top_k=_MAX_K)
    bm25_hits = [[h["id"] for h in row] for row in bm25_results]

    # [4/4] Dense search via Qdrant REST
    print(f"[4/4] Running dense (Qdrant) on {len(queries):,} queries (top-{_MAX_K})...")
    http = httpx.Client(base_url=QDRANT_URL, headers={"api-key": QDRANT_API_KEY})
    dense_hits = dense_search_batch(http, vectors)
    http.close()

    # Sanity check
    assert len(bm25_hits) == len(dense_hits) == len(relevant)

    # Compute recall at each pool size
    print()
    print(f"Pool Recall@K  (n={len(queries):,} queries, seed={_SAMPLE_SEED})")
    print(
        f"{'K':>5}  {'BM25':>8}  {'Dense':>8}  {'RRF':>8}  {'Union':>8}"
        f"   {'RRF gain':>9}  {'ceiling gap':>11}"
    )
    print("  " + "─" * 70)
    for k in _POOL_SIZES:
        r_bm25 = recall(bm25_hits, relevant, k)
        r_dense = recall(dense_hits, relevant, k)

        rrf_hits = [
            [h["id"] for h in rrf_fuse(_id_dicts(b_ids[:k]), _id_dicts(d_ids[:k]), top_k=k)]
            for b_ids, d_ids in zip(bm25_hits, dense_hits)
        ]
        r_rrf = recall(rrf_hits, relevant, k)

        # Set-union ceiling: GT is in BM25 top-k OR Dense top-k
        r_union = sum(
            1
            for b_ids, d_ids, rel in zip(bm25_hits, dense_hits, relevant)
            if rel in b_ids[:k] or rel in d_ids[:k]
        ) / len(queries)

        better_single = max(r_bm25, r_dense)
        rrf_gain = r_rrf - better_single        # vs. best single retriever
        ceiling_gap = r_union - r_rrf           # what RRF leaves on the table

        print(
            f"{k:>5}  {r_bm25:>8.4f}  {r_dense:>8.4f}  {r_rrf:>8.4f}  {r_union:>8.4f}"
            f"   {rrf_gain:>+9.4f}  {ceiling_gap:>+11.4f}"
        )

    print()
    print("Read:")
    print("  - 'RRF gain'    = Recall(RRF@K) − Recall(best of BM25/Dense @K).")
    print("                    Positive → fusion finds GT the better single list missed.")
    print("  - 'ceiling gap' = Recall(set-union@K) − Recall(RRF@K).")
    print("                    Positive → GT is in the union but RRF didn't surface it.")
    print("                    The reranker can recover this gap if the GT is anywhere")
    print("                    in the candidate set it sees.")


if __name__ == "__main__":
    main()
