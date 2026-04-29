"""
Eval harness: iterate all eval queries, call a retriever, aggregate metrics.

Usage:
    # Sanity check on 1 000 sampled queries (fast):
    python -m codesearch.eval.harness --retriever bm25 --max-queries 1000

    # Full eval:
    python -m codesearch.eval.harness --retriever bm25

Add new retriever names to _RETRIEVER_CHOICES and _build_retriever as M2/M3 land.
"""

from __future__ import annotations

import argparse
import random
import time
from typing import Protocol, runtime_checkable

from codesearch.config import SMOKE_TEST_SIZE
from codesearch.data import load_codesearch
from codesearch.eval.metrics import mean_ndcg, mean_recall, mean_reciprocal_rank
from codesearch.retrievers.bm25 import BM25Retriever
from codesearch.retrievers.dense import DenseRetriever

_RETRIEVER_CHOICES = ["bm25", "dense"]
_SAMPLE_SEED = 42


@runtime_checkable
class Retriever(Protocol):
    def retrieve(self, query: str, top_k: int) -> list[dict]: ...


def _build_retriever(name: str, corpus: list[dict]) -> Retriever:
    if name == "bm25":
        return BM25Retriever(corpus)
    if name == "dense":
        return DenseRetriever()
    # M3: elif name == "hybrid": return HybridRetriever(corpus)
    raise ValueError(f"Unknown retriever '{name}'. Available: {_RETRIEVER_CHOICES}")


def run_eval(retriever_name: str, max_queries: int | None = None) -> None:
    if SMOKE_TEST_SIZE != -1:
        print(
            f"[warn] SMOKE_TEST_SIZE={SMOKE_TEST_SIZE} — running on a subset.\n"
            f"       Set SMOKE_TEST_SIZE=-1 in .env for full eval."
        )

    corpus, queries = load_codesearch()
    retriever = _build_retriever(retriever_name, corpus)

    if max_queries and len(queries) > max_queries:
        random.seed(_SAMPLE_SEED)
        queries = random.sample(queries, max_queries)
        print(f"Sampled {max_queries:,} queries (seed={_SAMPLE_SEED}).")

    print(f"\nEvaluating '{retriever_name}' on {len(queries):,} queries...")
    t0 = time.perf_counter()

    query_results: list[tuple[list[str], str]] = []

    if hasattr(retriever, "retrieve_batch"):
        # Fast path: one vectorised call (BM25 via bm25s, and future dense batching).
        all_texts = [q["query"] for q in queries]
        all_hits = retriever.retrieve_batch(all_texts, top_k=100)
        for q, hits in zip(queries, all_hits):
            query_results.append(([h["id"] for h in hits], q["relevant_id"]))
    else:
        # Fallback: per-query loop (used by retrievers that don't support batching).
        from tqdm import tqdm
        for q in tqdm(queries, desc="Queries"):
            hits = retriever.retrieve(q["query"], top_k=100)
            query_results.append(([h["id"] for h in hits], q["relevant_id"]))

    elapsed = time.perf_counter() - t0
    ms_per_query = elapsed / len(queries) * 1000 if queries else 0.0

    mrr    = mean_reciprocal_rank(query_results, k=10)
    ndcg   = mean_ndcg(query_results, k=10)
    recall = mean_recall(query_results, k=100)

    sample_note = f" (sample n={max_queries})" if max_queries else ""
    print(f"\nRetriever : {retriever_name}")
    print(f"Corpus    : {len(corpus):,} docs")
    print(f"Queries   : {len(queries):,}{sample_note}")
    print()
    print(f"{'Metric':<12} {'Score':>8}")
    print("-" * 22)
    print(f"{'MRR@10':<12} {mrr:>8.4f}")
    print(f"{'nDCG@10':<12} {ndcg:>8.4f}")
    print(f"{'Recall@100':<12} {recall:>8.4f}")
    print()
    print(f"Elapsed : {elapsed:.1f}s  ({ms_per_query:.1f} ms/query)")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate a retriever on the CodeSearchNet Python split."
    )
    parser.add_argument(
        "--retriever",
        required=True,
        choices=_RETRIEVER_CHOICES,
        help="Which retriever to evaluate.",
    )
    parser.add_argument(
        "--max-queries",
        type=int,
        default=None,
        metavar="N",
        help="Evaluate on a random sample of N queries (default: all). "
             "Use 1000 for a fast sanity check before running the full eval.",
    )
    args = parser.parse_args()
    run_eval(args.retriever, max_queries=args.max_queries)


if __name__ == "__main__":
    main()
