"""
Hybrid retriever: Reciprocal Rank Fusion of BM25 + dense.

Composes (does not subclass) BM25Retriever and DenseRetriever. Pulls top-N
from each leg, fuses ranks via RRF, returns top-K by RRF score.

Why RRF rather than linear score blending:
    BM25 scores (log-IDF-weighted, unbounded) and cosine similarities (in [-1, 1])
    live on completely different scales. Linear combinations require per-corpus
    normalisation tuning that is fragile and corpus-specific. RRF sidesteps the
    problem entirely by operating on RANKS — both retrievers vote with positions,
    not magnitudes.

Why k=60:
    Standard since Cormack et al. (2009). Smaller k makes the score lean harder
    on rank-1; larger k flattens toward a uniform vote. Literature consensus is
    "don't tune it".

Per-result `score`, `bm25_rank`, `dense_rank` are exposed so downstream analysis
(e.g. twin-disambiguation in M3.3) can see WHY a doc rose or fell.
"""

from __future__ import annotations

from codesearch.retrievers.bm25 import BM25Retriever
from codesearch.retrievers.dense import DenseRetriever


_DEFAULT_POOL = 100
_DEFAULT_RRF_K = 60


def rrf_fuse(
    bm25_hits: list[dict],
    dense_hits: list[dict],
    top_k: int,
    rrf_k: int = _DEFAULT_RRF_K,
) -> list[dict]:
    """
    Fuse two ranked lists of hit-dicts into one via Reciprocal Rank Fusion.

        score(d) = 1/(rrf_k + rank_bm25(d)) + 1/(rrf_k + rank_dense(d))

    A doc missing from one list contributes 1/(rrf_k + (pool+1)) from that side,
    where pool = max(len(bm25_hits), len(dense_hits)). This penalises singletons
    relative to docs that appear in both lists, without dropping them entirely.

    When a doc appears in both, BM25's dict is preferred — it carries the full
    corpus payload (code, docstring, url, code_tokens), whereas dense hits are
    {id, score} only (Qdrant payload is doc_id-only by design; see dense.py).
    For dense-only hits the result dict will only carry id + score + the rank
    fields below; downstream callers that need code/docstring should look up
    by id in the in-memory corpus.

    Each returned dict carries the underlying payload plus:
        score      : RRF score (float)
        bm25_rank  : 1-indexed BM25 rank, or None if not in BM25's pool
        dense_rank : 1-indexed dense rank, or None if not in dense's pool
    """
    pool = max(len(bm25_hits), len(dense_hits))
    missing_rank = pool + 1

    bm25_by_id: dict[str, tuple[int, dict]] = {
        h["id"]: (rank, h) for rank, h in enumerate(bm25_hits, start=1)
    }
    dense_by_id: dict[str, tuple[int, dict]] = {
        h["id"]: (rank, h) for rank, h in enumerate(dense_hits, start=1)
    }

    fused: list[dict] = []
    for doc_id in set(bm25_by_id) | set(dense_by_id):
        b_rank = bm25_by_id[doc_id][0] if doc_id in bm25_by_id else None
        d_rank = dense_by_id[doc_id][0] if doc_id in dense_by_id else None

        score = 1.0 / (rrf_k + (b_rank if b_rank is not None else missing_rank))
        score += 1.0 / (rrf_k + (d_rank if d_rank is not None else missing_rank))

        payload = (
            bm25_by_id[doc_id][1] if doc_id in bm25_by_id else dense_by_id[doc_id][1]
        )
        fused.append(
            {
                **payload,
                "score": score,
                "bm25_rank": b_rank,
                "dense_rank": d_rank,
            }
        )

    fused.sort(key=lambda d: -d["score"])
    return fused[:top_k]


class HybridRetriever:
    def __init__(
        self,
        bm25: BM25Retriever,
        dense: DenseRetriever,
        rrf_k: int = _DEFAULT_RRF_K,
        pool_size: int = _DEFAULT_POOL,
    ) -> None:
        self.bm25 = bm25
        self.dense = dense
        self.rrf_k = rrf_k
        self.pool_size = pool_size

    def retrieve(self, query: str, top_k: int = 10) -> list[dict]:
        return self.retrieve_batch([query], top_k=top_k)[0]

    def retrieve_batch(
        self, queries: list[str], top_k: int = 100
    ) -> list[list[dict]]:
        assert top_k <= self.pool_size, (
            f"top_k={top_k} exceeds pool_size={self.pool_size}. "
            f"Raise pool_size or lower top_k."
        )
        n = len(queries)
        print(f"[hybrid 1/3] BM25 batch retrieve ({n:,} queries)...")
        bm25_results = self.bm25.retrieve_batch(queries, top_k=self.pool_size)

        print(f"[hybrid 2/3] Dense batch retrieve ({n:,} queries)...")
        dense_results = self.dense.retrieve_batch(queries, top_k=self.pool_size)

        print(f"[hybrid 3/3] RRF-fusing rank lists (k={self.rrf_k})...")
        return [
            rrf_fuse(b, d, top_k=top_k, rrf_k=self.rrf_k)
            for b, d in zip(bm25_results, dense_results)
        ]
