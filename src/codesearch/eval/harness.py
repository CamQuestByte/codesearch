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
from codesearch.retrievers.bm25_index import BM25Index
from codesearch.retrievers.dense import DenseRetriever
from codesearch.retrievers.hybrid import HybridRetriever
from codesearch.retrievers.reranker import CrossEncoderReranker

_RETRIEVER_CHOICES = ["bm25", "dense", "hybrid", "hybrid_rerank"]
_SAMPLE_SEED = 42
_BM25_CACHE_DIR = ".cache/bm25"


@runtime_checkable
class Retriever(Protocol):
    def retrieve(self, query: str, top_k: int) -> list[dict]: ...


def _load_bm25(corpus: list[dict]) -> BM25Retriever:
    """Prefer the on-disk index; otherwise build from corpus (~2-3 min on full)."""
    if BM25Index.exists(_BM25_CACHE_DIR):
        print(f"Loading cached BM25 index from {_BM25_CACHE_DIR}...")
        return BM25Retriever.from_cache(_BM25_CACHE_DIR)
    print(f"No BM25 cache at {_BM25_CACHE_DIR}; building from corpus.")
    return BM25Retriever(corpus)


def _build_retriever(name: str, corpus: list[dict]) -> Retriever:
    if name == "bm25":
        return _load_bm25(corpus)
    if name == "dense":
        return DenseRetriever()
    if name == "hybrid":
        return HybridRetriever(bm25=_load_bm25(corpus), dense=DenseRetriever())
    if name == "hybrid_rerank":
        hybrid = HybridRetriever(bm25=_load_bm25(corpus), dense=DenseRetriever())
        # code_tokens is already a joined string (see data.py::_make_doc). Safe
        # to feed to the CE — CodeSearchNet tokenises with the docstring stripped.
        corpus_lookup = {doc["id"]: doc["code_tokens"] for doc in corpus}
        return CrossEncoderReranker(base=hybrid, corpus_lookup=corpus_lookup)
    raise ValueError(f"Unknown retriever '{name}'. Available: {_RETRIEVER_CHOICES}")


def _needs_corpus(retriever_name: str) -> bool:
    """Whether the chosen retriever will actually read the corpus list.

    Dense never touches it (Qdrant has everything). BM25 / hybrid only touch it
    if the on-disk BM25 index is missing and has to be built from scratch.
    hybrid_rerank always needs it — the reranker builds a doc_id → code_tokens
    lookup dict for every candidate the CE has to score.
    """
    if retriever_name == "dense":
        return False
    if retriever_name == "hybrid_rerank":
        return True
    return not BM25Index.exists(_BM25_CACHE_DIR)


def _resolve_corpus_size(retriever: Retriever, in_memory_corpus: list[dict]) -> int:
    """Best-effort corpus size for the eval header line.

    When _needs_corpus() returned False, the in-memory list is empty but the
    underlying indexes still cover the full corpus — read the size from there.
    """
    if in_memory_corpus:
        return len(in_memory_corpus)
    if isinstance(retriever, CrossEncoderReranker):
        return _resolve_corpus_size(retriever.base, in_memory_corpus)
    if isinstance(retriever, HybridRetriever):
        return len(retriever.bm25._idx.corpus)
    if isinstance(retriever, BM25Retriever):
        return len(retriever._idx.corpus)
    if isinstance(retriever, DenseRetriever):
        try:
            return retriever.client.get_collection(retriever.collection).points_count
        except Exception:
            return -1
    return -1


def run_eval(retriever_name: str, max_queries: int | None = None) -> None:
    if SMOKE_TEST_SIZE != -1:
        print(
            f"[warn] SMOKE_TEST_SIZE={SMOKE_TEST_SIZE} — running on a subset.\n"
            f"       Set SMOKE_TEST_SIZE=-1 in .env for full eval."
        )

    if _needs_corpus(retriever_name):
        corpus, queries = load_codesearch()
    else:
        _, queries = load_codesearch(queries_only=True)
        corpus = []  # nothing downstream will read this
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
    corpus_size = _resolve_corpus_size(retriever, corpus)
    print(f"\nRetriever : {retriever_name}")
    print(f"Corpus    : {corpus_size:,} docs")
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
