"""
BM25 retriever backed by bm25s (vectorized sparse BM25).

Replaces rank_bm25 (pure-Python, O(N) per query) with bm25s, which builds a
sparse matrix index and scores all queries in a single batched call. On a 434k
corpus with 22k eval queries this cuts eval time from ~20h to ~5 min.

The public interface is unchanged from rank_bm25 — the app and eval harness
call the same retrieve() / retrieve_batch() methods.

Lucene analogy:
    BM25Retriever.__init__()      ≈ IndexWriter
    BM25Retriever.retrieve()      ≈ IndexSearcher.search()
    BM25Retriever.retrieve_batch()≈ IndexSearcher.search() over a query list

We index code (not docstrings) because CodeSearchNet queries are natural-language
descriptions and the task is to retrieve code — indexing docstrings makes BM25 a
near-exact lookup (query == indexed text) and inflates scores to ~0.96 MRR.
"""

from __future__ import annotations

import bm25s
import numpy as np


class BM25Retriever:
    def __init__(self, corpus: list[dict]) -> None:
        """
        Build the BM25 index over corpus docstrings.

        Args:
            corpus: list of dicts with at least 'id' and 'docstring' keys.
        """
        self.corpus = corpus

        print("Tokenizing corpus for BM25...")
        code = [doc["code_tokens"] for doc in corpus]
        corpus_tokens = bm25s.tokenize(code, stopwords=None, show_progress=True)

        self.index = bm25s.BM25()
        self.index.index(corpus_tokens)
        print(f"BM25 index built over {len(corpus):,} docs.")

    def retrieve(self, query: str, top_k: int = 10) -> list[dict]:
        """
        Retrieve top-k documents for a single query.

        Returns:
            List of dicts: [{id, code, docstring, url, score}, ...], score-descending.
        """
        return self.retrieve_batch([query], top_k=top_k)[0]

    def retrieve_batch(self, queries: list[str], top_k: int = 10) -> list[list[dict]]:
        """
        Retrieve top-k documents for a list of queries in one vectorized call.

        Returns:
            List of per-query result lists, each sorted by score descending.
        """
        k = min(top_k, len(self.corpus))
        query_tokens = bm25s.tokenize(queries, stopwords=None, show_progress=False)
        results, scores = self.index.retrieve(
            query_tokens, k=k, show_progress=True, leave_progress=False
        )
        # results: np.ndarray shape (n_queries, k) — doc indices into self.corpus
        # scores:  np.ndarray shape (n_queries, k) — BM25 scores, already sorted desc
        return [
            [
                {**self.corpus[int(results[i, j])], "score": float(scores[i, j])}
                for j in range(k)
            ]
            for i in range(len(queries))
        ]
