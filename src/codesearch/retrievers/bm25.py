"""
BM25 retriever backed by bm25s (vectorized sparse BM25).

Replaces rank_bm25 (pure-Python, O(N) per query) with bm25s, which builds a
sparse matrix index and scores all queries in a single batched call. On a 434k
corpus with 22k eval queries this cuts eval time from ~20h to ~5 min.

The public interface is unchanged from rank_bm25 — the app and eval harness
call the same retrieve() / retrieve_batch() methods.

Lucene analogy:
    BM25Retriever       ≈ IndexSearcher (the scoring/query layer)
    BM25Index           ≈ Directory + IndexReader (the persistable data; see bm25_index.py)
    BM25Retriever.retrieve()      ≈ IndexSearcher.search()
    BM25Retriever.retrieve_batch()≈ IndexSearcher.search() over a query list

We index code (not docstrings) because CodeSearchNet queries are natural-language
descriptions and the task is to retrieve code — indexing docstrings makes BM25 a
near-exact lookup (query == indexed text) and inflates scores to ~0.96 MRR.
"""

from __future__ import annotations

from typing import Union

import bm25s

from codesearch.retrievers.bm25_index import BM25Index


class BM25Retriever:
    def __init__(self, corpus_or_index: Union[list[dict], BM25Index]) -> None:
        """
        Construct a retriever from either a raw corpus (builds the index now)
        or an already-built BM25Index (e.g. loaded from disk).

        Args:
            corpus_or_index:
                - list[dict] of docs with 'code_tokens' / 'id' / etc.
                  → BM25Index.build() runs (a few minutes on the full corpus)
                - BM25Index instance
                  → reused as-is, no rebuild
        """
        if isinstance(corpus_or_index, BM25Index):
            self._idx = corpus_or_index
        else:
            self._idx = BM25Index.build(corpus_or_index)

    @classmethod
    def from_cache(cls, cache_dir: str) -> "BM25Retriever":
        """Shortcut for BM25Retriever(BM25Index.load(cache_dir))."""
        return cls(BM25Index.load(cache_dir))

    # Backwards-compatible attribute access — callers expect .corpus / .index.
    @property
    def corpus(self) -> list[dict]:
        return self._idx.corpus

    @property
    def index(self) -> bm25s.BM25:
        return self._idx.index

    def retrieve(self, query: str, top_k: int = 10) -> list[dict]:
        """Retrieve top-k documents for a single query."""
        return self.retrieve_batch([query], top_k=top_k)[0]

    def retrieve_batch(self, queries: list[str], top_k: int = 10) -> list[list[dict]]:
        """
        Retrieve top-k documents for a list of queries in one vectorized call.

        Returns:
            List of per-query result lists, each sorted by score descending.
        """
        corpus = self._idx.corpus
        k = min(top_k, len(corpus))
        query_tokens = bm25s.tokenize(queries, stopwords=None, show_progress=False)
        results, scores = self._idx.index.retrieve(
            query_tokens, k=k, show_progress=True, leave_progress=False
        )
        # results: np.ndarray shape (n_queries, k) — doc indices into corpus
        # scores:  np.ndarray shape (n_queries, k) — BM25 scores, already sorted desc
        return [
            [
                {**corpus[int(results[i, j])], "score": float(scores[i, j])}
                for j in range(k)
            ]
            for i in range(len(queries))
        ]
