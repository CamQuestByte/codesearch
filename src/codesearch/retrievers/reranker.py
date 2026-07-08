"""
Cross-encoder reranker: reorders a base retriever's top-K candidates.

Wraps any object exposing `retrieve_batch()`. Pulls top-`pool_size` from the
base, scores each (query, candidate_text) pair through a cross-encoder,
returns the top_k highest-scored candidates per query.

Why a cross-encoder here:
    RRF (M3.1) fattens the top-100 pool but rarely lifts anything to rank-1 —
    it operates on ranks only, can't see content. A cross-encoder reads both
    query and candidate through the same attention stack and outputs a direct
    relevance score. That's the piece meant to move MRR@10.

Why `ms-marco-MiniLM-L-6-v2` as the baseline:
    22M params, ~1000 pairs/sec on CPU. Trained on MS MARCO passage ranking,
    so out-of-domain for NL→code (same domain gap as the base MiniLM embedder,
    which still works). Establishes the off-the-shelf reranker line; code
    specific finetuning is M5 territory.

Runtime:
    Full eval = 22k queries × 100 pool = 2.2M forward passes ≈ 30 min on CPU.
    For iteration, run with `--max-queries 2000` (~3 min). No (query, doc_id)
    score cache yet — add one only when we start ablating pool_size / model.

Docstring-leakage note:
    The CodeSearchNet eval query IS the target function's docstring. The
    `code_tokens` field in the loaded corpus (data.py::_make_doc) is
    docstring-stripped by CSN's own tokenizer, so feeding it as candidate
    text is safe. Feeding raw `code` (which includes the docstring) would
    trivially match query→doc and produce meaningless metrics. See
    [[todo-embed-raw-code]] for an M5 alternative (AST-stripped
    `whole_func_string`) that keeps richer structural signal.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from sentence_transformers import CrossEncoder
from tqdm import tqdm


_DEFAULT_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
_DEFAULT_POOL = 100
# Outer chunk: how many queries' worth of pairs to hold in Python at once.
# Bounds peak pair-list memory (200 queries × 100 pool = ~20k tuples).
_QUERY_CHUNK = 200
# Inner chunk: pairs per CE.predict call. Chosen so the tqdm bar advances
# every ~1s on CPU (~500-1000 pairs/s) without spamming updates. Must be
# a multiple of batch_size so predict never runs a partial internal batch.
_PAIR_SUBCHUNK = 512


@runtime_checkable
class BaseRetriever(Protocol):
    def retrieve_batch(
        self, queries: list[str], top_k: int
    ) -> list[list[dict]]: ...


class CrossEncoderReranker:
    def __init__(
        self,
        base: BaseRetriever,
        corpus_lookup: dict[str, str],
        pool_size: int = _DEFAULT_POOL,
        model_name: str = _DEFAULT_MODEL,
        batch_size: int = 32,
    ) -> None:
        self.base = base
        self.corpus_lookup = corpus_lookup
        self.pool_size = pool_size
        self.batch_size = batch_size
        print(f"Loading cross-encoder: {model_name}")
        self.model = CrossEncoder(model_name, max_length=512)

    def retrieve(self, query: str, top_k: int = 10) -> list[dict]:
        return self.retrieve_batch([query], top_k=top_k)[0]

    def retrieve_batch(
        self, queries: list[str], top_k: int = 10
    ) -> list[list[dict]]:
        assert top_k <= self.pool_size, (
            f"top_k={top_k} exceeds pool_size={self.pool_size}. "
            f"Raise pool_size or lower top_k."
        )

        print(
            f"[rerank 1/2] base retrieve ({len(queries):,} queries, "
            f"pool={self.pool_size})..."
        )
        pool = self.base.retrieve_batch(queries, top_k=self.pool_size)

        total_pairs = sum(len(hs) for hs in pool)
        print(f"[rerank 2/2] cross-encoder scoring ({total_pairs:,} pairs)...")

        out: list[list[dict]] = []
        bar = tqdm(total=total_pairs, desc="[rerank] pairs", unit="pair")
        try:
            for start in range(0, len(queries), _QUERY_CHUNK):
                qs = queries[start:start + _QUERY_CHUNK]
                hits = pool[start:start + _QUERY_CHUNK]

                # Flat pair list for this chunk; per-query slice via offsets.
                # KeyError on a missing id is intentional — every base-retriever
                # hit id must be present in the corpus lookup.
                pairs: list[tuple[str, str]] = []
                offsets: list[int] = [0]
                for q, hs in zip(qs, hits):
                    pairs.extend((q, self.corpus_lookup[h["id"]]) for h in hs)
                    offsets.append(len(pairs))

                # Score in sub-chunks so the bar advances between predict calls.
                # One giant predict on the full chunk would freeze the bar for
                # its whole duration (~15-30s+ on CPU).
                scores: list[float] = []
                for i in range(0, len(pairs), _PAIR_SUBCHUNK):
                    j = min(i + _PAIR_SUBCHUNK, len(pairs))
                    scores.extend(
                        self.model.predict(
                            pairs[i:j],
                            batch_size=self.batch_size,
                            show_progress_bar=False,
                        )
                    )
                    bar.update(j - i)

                for i, hs in enumerate(hits):
                    sl = scores[offsets[i]:offsets[i + 1]]
                    order = sorted(range(len(hs)), key=lambda k: -sl[k])
                    out.append(
                        [
                            {
                                **hs[k],
                                "pre_rerank_score": hs[k]["score"],
                                "score": float(sl[k]),
                            }
                            for k in order[:top_k]
                        ]
                    )
        finally:
            bar.close()

        return out
