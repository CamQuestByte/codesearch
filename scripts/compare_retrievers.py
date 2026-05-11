"""
Find queries that illustrate where dense beats BM25, where BM25 beats dense,
and where both nail it.

M2 done-criterion: be able to explain one query per category with a concrete
reason, not just "dense is better at semantics." This script surfaces the
candidates; the writeup is for the human.

Pipeline:
  1. Load cached query vectors (no re-embed) — see scripts/cache_query_vectors.py
  2. Sample to --max-queries
  3. Load full corpus, build BM25 index
  4. Run BM25 batch on sample
  5. Run dense batch (Qdrant via httpx) on sample using cached vectors
  6. For each query: rank of relevant doc in BM25 and in dense (or None if >100)
  7. Classify and print top --top-each examples per category with code snippets

Usage:
    uv run python scripts/cache_query_vectors.py     # one-time, if not done
    uv run python scripts/compare_retrievers.py
    uv run python scripts/compare_retrievers.py --max-queries 5000 --top-each 5
"""

from __future__ import annotations

import argparse
import os
import pickle
import random
import sys
import textwrap

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

_SAMPLE_SEED = 42
_TOP_K = 100
_SEARCH_BATCH = 50
CACHE_DIR = ".cache"
BM25_CACHE_DIR = ".cache/bm25"

# Classification thresholds for an "X wins" example
_RANK_GOOD = 5      # winner placed relevant doc at rank <= this
_RANK_BAD = 20      # loser placed it deeper than this (or missed it entirely)


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
    """Returns one list of ranked doc_ids per query."""
    hits_all: list[list[str]] = []
    n_batches = (len(vectors) + _SEARCH_BATCH - 1) // _SEARCH_BATCH
    for i in tqdm(range(0, len(vectors), _SEARCH_BATCH), total=n_batches, desc="Dense search"):
        chunk = vectors[i : i + _SEARCH_BATCH]
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
        for resp in r.json()["result"]:
            hits_all.append([p["payload"]["doc_id"] for p in resp["points"]])
    return hits_all


def rank_of(hit_ids: list[str], relevant_id: str) -> int | None:
    for i, h in enumerate(hit_ids, start=1):
        if h == relevant_id:
            return i
    return None


def fmt_rank(r: int | None) -> str:
    return f"#{r}" if r is not None else ">100"


def _snippet(text: str | None, n_lines: int = 4, max_chars: int = 280) -> str:
    if not text:
        return "(empty)"
    body = "\n".join(text.strip().splitlines()[:n_lines])
    if len(body) > max_chars:
        body = body[:max_chars].rstrip() + " ..."
    return body


def print_example(
    idx: int,
    category: str,
    q: dict,
    gt_doc: dict | None,
    bm25_rank: int | None,
    dense_rank: int | None,
    bm25_top1_doc: dict | None,
    dense_top1_doc: dict | None,
) -> None:
    sep = "─" * 78
    print(f"\n{sep}")
    print(f"  {category} — example #{idx}")
    print(sep)
    print(f"Query: {q['query'].strip()[:240]}")
    print()
    print(f"  Ground-truth doc:  {q['relevant_id']}")
    print(f"    BM25 rank: {fmt_rank(bm25_rank):>5}     Dense rank: {fmt_rank(dense_rank):>5}")
    if gt_doc:
        print(f"    GT docstring: {(gt_doc.get('docstring') or '').strip()[:160]}")
        print(f"    GT code:")
        print(textwrap.indent(_snippet(gt_doc.get("code")), "      "))
    print()
    if bm25_top1_doc is not None:
        print(f"  BM25 top-1: {bm25_top1_doc.get('id', '?')}")
        print(textwrap.indent(_snippet(bm25_top1_doc.get("code")), "      "))
        print()
    if dense_top1_doc is not None:
        print(f"  Dense top-1: {dense_top1_doc.get('id', '?')}")
        print(textwrap.indent(_snippet(dense_top1_doc.get("code")), "      "))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Find dense-vs-BM25 illustrative queries (M2 concept-checkpoint)."
    )
    parser.add_argument(
        "--max-queries",
        type=int,
        default=2000,
        help="Sample size (default: 2000 — plenty to surface clear examples).",
    )
    parser.add_argument(
        "--top-each",
        type=int,
        default=3,
        help="Examples to print per category (default: 3).",
    )
    args = parser.parse_args()

    # [1/5] Cached queries + vectors
    print("[1/5] Loading cached query vectors...")
    queries, vectors = load_cached()
    print(f"  Loaded {len(queries):,} queries.")

    if args.max_queries and len(queries) > args.max_queries:
        random.seed(_SAMPLE_SEED)
        idx = random.sample(range(len(queries)), args.max_queries)
        queries = [queries[i] for i in idx]
        vectors = [vectors[i] for i in idx]
        print(f"  Sampled {len(queries):,} queries (seed={_SAMPLE_SEED}).")

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
    corpus_by_id = {doc["id"]: doc for doc in bm25.corpus}

    # [3/4] BM25 search
    print(f"[3/4] Running BM25 on {len(queries):,} queries...")
    bm25_results = bm25.retrieve_batch([q["query"] for q in queries], top_k=_TOP_K)
    bm25_hits = [[h["id"] for h in row] for row in bm25_results]

    # [4/4] Dense search via Qdrant REST
    print(f"[4/4] Running dense (Qdrant) on {len(queries):,} queries...")
    http = httpx.Client(
        base_url=QDRANT_URL,
        headers={"api-key": QDRANT_API_KEY},
    )
    dense_hits = dense_search_batch(http, vectors)
    http.close()

    # Compute ranks
    annotated = []
    for q, b_ids, d_ids in zip(queries, bm25_hits, dense_hits):
        rel = q["relevant_id"]
        annotated.append(
            {
                "query": q,
                "bm25_rank": rank_of(b_ids, rel),
                "dense_rank": rank_of(d_ids, rel),
                "bm25_top1": b_ids[0] if b_ids else None,
                "dense_top1": d_ids[0] if d_ids else None,
            }
        )

    # Classify
    def well(r):
        return r is not None and r <= _RANK_GOOD

    def badly(r):
        return r is None or r > _RANK_BAD

    dense_wins = [r for r in annotated if well(r["dense_rank"]) and badly(r["bm25_rank"])]
    bm25_wins = [r for r in annotated if well(r["bm25_rank"]) and badly(r["dense_rank"])]
    ties = [r for r in annotated if r["bm25_rank"] == 1 and r["dense_rank"] == 1]

    # Sort by margin so the most striking cases come first
    dense_wins.sort(key=lambda r: -(r["bm25_rank"] or 9999))
    bm25_wins.sort(key=lambda r: -(r["dense_rank"] or 9999))
    random.seed(_SAMPLE_SEED)
    if ties:
        random.shuffle(ties)

    # Summary
    n = len(annotated)
    print(
        f"\nClassification on {n:,} queries "
        f"(winner rank<={_RANK_GOOD}, loser rank>{_RANK_BAD} or missing):"
    )
    print(f"  Dense wins  : {len(dense_wins):>5}  ({len(dense_wins)/n:.1%})")
    print(f"  BM25 wins   : {len(bm25_wins):>5}  ({len(bm25_wins)/n:.1%})")
    print(f"  Both at #1  : {len(ties):>5}  ({len(ties)/n:.1%})")

    def doc(doc_id):
        return corpus_by_id.get(doc_id)

    # Output
    print("\n" + "=" * 78)
    print(f"  DENSE WINS  (dense top-{_RANK_GOOD}, BM25 missed)")
    print("=" * 78)
    for i, r in enumerate(dense_wins[: args.top_each], start=1):
        print_example(
            i,
            "Dense win",
            r["query"],
            doc(r["query"]["relevant_id"]),
            r["bm25_rank"],
            r["dense_rank"],
            doc(r["bm25_top1"]),
            doc(r["dense_top1"]),
        )

    print("\n\n" + "=" * 78)
    print(f"  BM25 WINS  (BM25 top-{_RANK_GOOD}, dense missed)")
    print("=" * 78)
    for i, r in enumerate(bm25_wins[: args.top_each], start=1):
        print_example(
            i,
            "BM25 win",
            r["query"],
            doc(r["query"]["relevant_id"]),
            r["bm25_rank"],
            r["dense_rank"],
            doc(r["bm25_top1"]),
            doc(r["dense_top1"]),
        )

    print("\n\n" + "=" * 78)
    print("  TIES  (both retrievers placed relevant doc at rank #1)")
    print("=" * 78)
    for i, r in enumerate(ties[: args.top_each], start=1):
        print_example(
            i,
            "Tie",
            r["query"],
            doc(r["query"]["relevant_id"]),
            r["bm25_rank"],
            r["dense_rank"],
            doc(r["bm25_top1"]),
            doc(r["dense_top1"]),
        )


if __name__ == "__main__":
    main()
