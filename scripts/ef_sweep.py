"""
HNSW ef_search sweep — M2 concept checkpoint.

Reads pre-cached query vectors from disk (run scripts/cache_query_vectors.py
first) and sweeps the HNSW beam width to chart the recall/latency tradeoff.

For each ef we report:
    Recall@100, MRR@10  — accuracy
    server ms/q         — Qdrant's reported time per query (from response.time)
    wall ms/q           — client-side wall time per query (server + network RTT)

The delta between server and wall ms/q is how much of "ms per query" is
actually network round-trip to Qdrant Cloud, not HNSW work.

Usage:
    uv run python scripts/cache_query_vectors.py            # one-time
    uv run python scripts/ef_sweep.py
    uv run python scripts/ef_sweep.py --ef-values 16,32,64,128,256,512
    uv run python scripts/ef_sweep.py --max-queries 1000
"""

from __future__ import annotations

import argparse
import os
import pickle
import random
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

import httpx
from tqdm import tqdm

from codesearch.config import (
    EMBEDDING_MODEL,
    QDRANT_API_KEY,
    QDRANT_COLLECTION,
    QDRANT_URL,
)
from codesearch.eval.metrics import mean_recall, mean_reciprocal_rank

_SAMPLE_SEED = 42       # match eval/harness.py
_SEARCH_BATCH = 50      # queries per HTTP request to Qdrant (matches dense.py)
_TOP_K = 100
CACHE_DIR = ".cache"


def cache_path(model_name: str) -> str:
    safe = model_name.replace("/", "_")
    return os.path.join(CACHE_DIR, f"query_vectors_{safe}.pkl")


def load_cached_queries() -> tuple[list[dict], list[list[float]]]:
    path = cache_path(EMBEDDING_MODEL)
    if not os.path.exists(path):
        sys.exit(
            f"[error] Cache not found at {path}.\n"
            f"        Run: uv run python scripts/cache_query_vectors.py"
        )
    print(f"[1/3] Loading cached query vectors from {path}...")
    with open(path, "rb") as f:
        data = pickle.load(f)
    if data["model"] != EMBEDDING_MODEL:
        sys.exit(
            f"[error] Cache model mismatch:\n"
            f"        cache built for: {data['model']!r}\n"
            f"        current model:   {EMBEDDING_MODEL!r}\n"
            f"        Run: uv run python scripts/cache_query_vectors.py --recompute"
        )
    queries = data["queries"]
    vectors = data["vectors"].tolist()
    print(f"  Loaded {len(queries):,} queries and their vectors.")
    return queries, vectors


def search_batch(
    http: httpx.Client,
    chunk: list[list[float]],
    ef: int,
) -> tuple[list[list[str]], float]:
    """One HTTP call to Qdrant for a chunk of <=_SEARCH_BATCH queries.
    Returns (hits_per_query, server_time_seconds)."""
    payload = {
        "searches": [
            {
                "query": qv,
                "limit": _TOP_K,
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
    body = r.json()
    server_time = float(body.get("time", 0.0))
    hits = [
        [p["payload"]["doc_id"] for p in resp["points"]]
        for resp in body["result"]
    ]
    return hits, server_time


def sweep_one_ef(
    http: httpx.Client,
    query_vectors: list[list[float]],
    ef: int,
) -> tuple[list[list[str]], float, float]:
    """Run all batches at a given ef. Returns (hits, wall_seconds, server_seconds)."""
    all_hits: list[list[str]] = []
    server_total = 0.0
    n_batches = (len(query_vectors) + _SEARCH_BATCH - 1) // _SEARCH_BATCH

    wall_t0 = time.perf_counter()
    for i in tqdm(
        range(0, len(query_vectors), _SEARCH_BATCH),
        total=n_batches,
        desc=f"  ef={ef:<4}",
        leave=False,
    ):
        chunk = query_vectors[i : i + _SEARCH_BATCH]
        hits, server_t = search_batch(http, chunk, ef)
        all_hits.extend(hits)
        server_total += server_t
    wall_elapsed = time.perf_counter() - wall_t0
    return all_hits, wall_elapsed, server_total


def run_sweep(ef_values: list[int], max_queries: int) -> None:
    # ------------------------------------------------------------------
    # [1] Load cached queries + vectors
    # ------------------------------------------------------------------
    queries, vectors = load_cached_queries()

    if max_queries and len(queries) > max_queries:
        random.seed(_SAMPLE_SEED)
        idx = random.sample(range(len(queries)), max_queries)
        queries = [queries[i] for i in idx]
        vectors = [vectors[i] for i in idx]
        print(f"  Sampled {len(queries):,} queries (seed={_SAMPLE_SEED}).")

    # ------------------------------------------------------------------
    # [2] Verify Qdrant collection is populated
    # ------------------------------------------------------------------
    print(f"[2/3] Connecting to Qdrant at {QDRANT_URL}")
    http = httpx.Client(
        base_url=QDRANT_URL,
        headers={"api-key": QDRANT_API_KEY},
    )
    info = http.get(f"/collections/{QDRANT_COLLECTION}").json()
    points_count = info.get("result", {}).get("points_count", 0)
    if not points_count:
        sys.exit(
            f"[error] Collection '{QDRANT_COLLECTION}' is empty.\n"
            f"        Run: uv run python scripts/index_corpus.py"
        )
    print(f"  Collection '{QDRANT_COLLECTION}' has {points_count:,} vectors.")

    # ------------------------------------------------------------------
    # [3] Warmup, then sweep
    # ------------------------------------------------------------------
    print(f"[3/3] Warming up (1 batch at ef={ef_values[0]})...")
    search_batch(http, vectors[: min(_SEARCH_BATCH, len(vectors))], ef_values[0])

    print(f"      Sweeping ef across {ef_values}...")
    relevant_ids = [q["relevant_id"] for q in queries]
    rows: list[tuple[int, float, float, float, float]] = []

    for ef in tqdm(ef_values, desc="Sweep"):
        hits, wall_s, server_s = sweep_one_ef(http, vectors, ef)
        query_results = list(zip(hits, relevant_ids))
        mrr = mean_reciprocal_rank(query_results, k=10)
        recall = mean_recall(query_results, k=100)
        n = len(queries)
        rows.append((ef, mrr, recall, server_s / n * 1000, wall_s / n * 1000))

    http.close()

    # ------------------------------------------------------------------
    # Results table
    # ------------------------------------------------------------------
    print()
    print(f"Dense retrieval HNSW ef_search sweep — n={len(queries)} queries, top-{_TOP_K}")
    header = f"{'ef':>6} {'MRR@10':>9} {'Recall@100':>12} {'server ms/q':>14} {'wall ms/q':>12}"
    print(header)
    print("-" * len(header))
    for ef, mrr, recall, server_ms, wall_ms in rows:
        print(f"{ef:>6} {mrr:>9.4f} {recall:>12.4f} {server_ms:>14.2f} {wall_ms:>12.2f}")
    print()
    print("server ms/q = Qdrant's reported processing time per query (HNSW work only)")
    print("wall ms/q   = client-side wall time per query (server time + network RTT)")
    print("(query embedding is cached and excluded from both)")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Sweep HNSW ef_search on the dense Qdrant index."
    )
    parser.add_argument(
        "--ef-values",
        default="32,128,512",
        help="Comma-separated ef_search values to sweep (default: 32,128,512 per PLAN.md).",
    )
    parser.add_argument(
        "--max-queries",
        type=int,
        default=500,
        help="Sample size (default: 500 — ~±2pp stderr on Recall@100, fast runtime).",
    )
    args = parser.parse_args()

    ef_values = [int(x) for x in args.ef_values.split(",")]
    run_sweep(ef_values, args.max_queries)


if __name__ == "__main__":
    main()
