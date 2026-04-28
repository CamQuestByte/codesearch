"""
Retrieval evaluation metrics — all pure functions, no side effects.

Per-query functions accept:
    retrieved_ids : list[str]  — ranked doc IDs, best-first
    relevant_id   : str        — the single ground-truth doc ID

Aggregate functions accept a list of (retrieved_ids, relevant_id) pairs
and return the mean score across all queries.
"""

from __future__ import annotations

import math


# ---------------------------------------------------------------------------
# Per-query
# ---------------------------------------------------------------------------

def reciprocal_rank(retrieved_ids: list[str], relevant_id: str, k: int = 10) -> float:
    """1/rank if relevant doc is in top-k, else 0."""
    for rank, doc_id in enumerate(retrieved_ids[:k], start=1):
        if doc_id == relevant_id:
            return 1.0 / rank
    return 0.0


def ndcg(retrieved_ids: list[str], relevant_id: str, k: int = 10) -> float:
    """nDCG@k with binary single-relevant judgments.

    With one relevant doc, IDCG is always 1/log2(2) = 1.0, so
    nDCG reduces to DCG: 1/log2(rank + 1) if found in top-k, else 0.
    """
    for rank, doc_id in enumerate(retrieved_ids[:k], start=1):
        if doc_id == relevant_id:
            return 1.0 / math.log2(rank + 1)
    return 0.0


def recall_at_k(retrieved_ids: list[str], relevant_id: str, k: int = 100) -> float:
    """1.0 if relevant doc appears in top-k results, else 0.0."""
    return 1.0 if relevant_id in retrieved_ids[:k] else 0.0


# ---------------------------------------------------------------------------
# Aggregate (mean over all queries)
# ---------------------------------------------------------------------------

QueryResults = list[tuple[list[str], str]]  # [(retrieved_ids, relevant_id), ...]


def mean_reciprocal_rank(query_results: QueryResults, k: int = 10) -> float:
    """MRR@k — CodeSearchNet canonical metric."""
    if not query_results:
        return 0.0
    return sum(reciprocal_rank(r, rel, k) for r, rel in query_results) / len(query_results)


def mean_ndcg(query_results: QueryResults, k: int = 10) -> float:
    """Mean nDCG@k — standard BEIR metric."""
    if not query_results:
        return 0.0
    return sum(ndcg(r, rel, k) for r, rel in query_results) / len(query_results)


def mean_recall(query_results: QueryResults, k: int = 100) -> float:
    """Mean Recall@k — primary retriever health metric."""
    if not query_results:
        return 0.0
    return sum(recall_at_k(r, rel, k) for r, rel in query_results) / len(query_results)


# ---------------------------------------------------------------------------
# Sanity checks (run with: python -m codesearch.eval.metrics)
# ---------------------------------------------------------------------------

def _run_sanity_checks() -> None:
    # --- reciprocal_rank ---
    assert reciprocal_rank(["a", "b", "c"], "a") == 1.0,        "rank=1 should give MRR=1.0"
    assert reciprocal_rank(["b", "a", "c"], "a") == 0.5,        "rank=2 should give MRR=0.5"
    assert reciprocal_rank(["x"] * 10 + ["a"], "a") == 0.0,     "rank=11 should give MRR=0 (beyond k=10)"
    assert reciprocal_rank(["b", "c"], "a") == 0.0,             "not found should give MRR=0"
    assert reciprocal_rank(["b", "a"], "a", k=1) == 0.0,        "rank=2 beyond k=1 should give 0"

    # --- ndcg ---
    assert ndcg(["a", "b", "c"], "a") == 1.0,                              "rank=1 nDCG should be 1.0"
    assert abs(ndcg(["b", "a", "c"], "a") - 1 / math.log2(3)) < 1e-12,    "rank=2 nDCG should be 1/log2(3)"
    assert ndcg(["x"] * 10 + ["a"], "a") == 0.0,                           "rank=11 nDCG should be 0"

    # --- recall_at_k ---
    assert recall_at_k(["a", "b"], "a") == 1.0,                 "found in top-100 should be 1.0"
    assert recall_at_k(["b", "c"], "a") == 0.0,                 "not found should be 0.0"
    assert recall_at_k(["b", "a"], "a", k=1) == 0.0,            "rank=2 beyond k=1 should be 0.0"

    # --- aggregates ---
    results = [
        (["a", "b"], "a"),   # RR=1.0, nDCG=1.0, R@100=1.0
        (["b", "a"], "a"),   # RR=0.5, nDCG=1/log2(3), R@100=1.0
        (["b", "c"], "a"),   # RR=0.0, nDCG=0.0, R@100=0.0
    ]
    assert abs(mean_reciprocal_rank(results) - (1.0 + 0.5 + 0.0) / 3) < 1e-12
    assert mean_recall(results) == 2 / 3

    print("All sanity checks passed.")


if __name__ == "__main__":
    _run_sanity_checks()
