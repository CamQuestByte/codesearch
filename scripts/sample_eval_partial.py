"""M5 sub-experiment B — early directional read from a PARTIAL golds-first index.

The full UniXcoder index runs golds-first (index_corpus.py GOLDS_FIRST=1), so the
first ~22k embedded docs are every eval query's gold target; docs after that are
train-split distractors whose pool grows as indexing proceeds. This script does a
fair, exact brute-force cosine A/B over the SAME candidate pool for both models,
reading vectors straight from the .cache/*.npy files — no Qdrant, no HNSW recall
loss, no CPU contention with the running indexer.

Pool at run time = the first P = unixcoder embed-progress docs (golds-first order).
UniXcoder doc vectors come from its cache[0:P]; MiniLM doc vectors are pulled from
the full MiniLM cache for the SAME doc set (mapped by doc_id). Queries evaluated =
those whose gold is within the pool. Metrics: MRR@10 / nDCG@10 / Recall@100.

Usage:
    .venv/bin/python scripts/sample_eval_partial.py            # eval at current P
    .venv/bin/python scripts/sample_eval_partial.py --min-docs 44000   # require P first
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, "src")
from codesearch.data import load_codesearch  # noqa: E402
from codesearch.eval.metrics import (  # noqa: E402
    mean_ndcg,
    mean_recall,
    mean_reciprocal_rank,
)
from codesearch.embedding import load_encoder  # noqa: E402

CACHE = ".cache"
UNIX_MODEL = "microsoft/unixcoder-base"
MINILM_MODEL = "all-MiniLM-L6-v2"
UNIX_NPY = f"{CACHE}/corpus_vectors_microsoft_unixcoder-base.npy"
UNIX_PROG = f"{CACHE}/corpus_vectors_microsoft_unixcoder-base.progress"
MINILM_NPY = f"{CACHE}/corpus_vectors_all-MiniLM-L6-v2.npy"
UNIX_DIM, MINILM_DIM = 768, 384


def _read_progress(path: str) -> int:
    with open(path) as f:
        return int(f.read().strip() or "0")


def _golds_first_corpus(corpus, queries):
    """Reproduce index_corpus.py's GOLDS_FIRST reorder exactly."""
    gold_ids = {q["relevant_id"] for q in queries}
    golds = [d for d in corpus if d["id"] in gold_ids]
    rest = [d for d in corpus if d["id"] not in gold_ids]
    return golds + rest


def _topk_ids(query_vecs, doc_vecs, doc_ids, k=100, chunk=512):
    """Brute-force cosine (vectors are pre-normalized): top-k doc_ids per query."""
    out = []
    for i in range(0, len(query_vecs), chunk):
        sims = query_vecs[i : i + chunk] @ doc_vecs.T           # (b, P)
        kk = min(k, sims.shape[1])
        part = np.argpartition(-sims, kk - 1, axis=1)[:, :kk]
        for r in range(sims.shape[0]):
            order = part[r][np.argsort(-sims[r, part[r]])]
            out.append([doc_ids[j] for j in order])
    return out


def _eval_model(name, npy_path, dim, doc_rows, query_texts, relevant_ids, encoder):
    """doc_rows: np.ndarray of cache-row indices for the pool docs (in pool order)."""
    mm = np.memmap(npy_path, dtype=np.float32, mode="r").reshape(-1, dim)
    doc_vecs = np.ascontiguousarray(mm[doc_rows])                # (P, dim)
    print(f"[{name}] encoding {len(query_texts):,} queries...", flush=True)
    qv = encoder.encode(query_texts, normalize_embeddings=True, batch_size=128)
    qv = np.asarray(qv, dtype=np.float32)
    retrieved = _topk_ids(qv, doc_vecs, POOL_IDS, k=100)
    qr = list(zip(retrieved, relevant_ids))
    return (
        mean_reciprocal_rank(qr, k=10),
        mean_ndcg(qr, k=10),
        mean_recall(qr, k=100),
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-docs", type=int, default=0,
                    help="Require at least this many embedded docs before evaluating.")
    args = ap.parse_args()

    P = _read_progress(UNIX_PROG)
    print(f"UniXcoder embed-progress P = {P:,} docs")
    if P < args.min_docs:
        print(f"Not enough yet (need {args.min_docs:,}); exiting.")
        return 0

    corpus, queries = load_codesearch(n=-1)
    orig_row = {d["id"]: i for i, d in enumerate(corpus)}        # doc_id -> MiniLM cache row
    gf = _golds_first_corpus(corpus, queries)                    # golds-first (== unixcoder cache order)
    pool = gf[:P]
    global POOL_IDS
    POOL_IDS = [d["id"] for d in pool]
    pool_id_set = set(POOL_IDS)

    # rows into each cache for the pool docs, in pool order
    unix_rows = np.arange(P)                                     # unixcoder cache is already golds-first
    minilm_rows = np.array([orig_row[i] for i in POOL_IDS])      # map to original-order MiniLM cache

    subset = [q for q in queries if q["relevant_id"] in pool_id_set]
    texts = [q["query"] for q in subset]
    rel = [q["relevant_id"] for q in subset]
    print(f"Pool: {P:,} docs  |  eval queries (gold in pool): {len(subset):,}")
    if not subset:
        print("No queries have their gold in the pool yet — wait for more golds.")
        return 0

    unix_enc = load_encoder(UNIX_MODEL)
    minilm_enc = load_encoder(MINILM_MODEL)

    u_mrr, u_ndcg, u_rec = _eval_model("unixcoder", UNIX_NPY, UNIX_DIM, unix_rows, texts, rel, unix_enc)
    m_mrr, m_ndcg, m_rec = _eval_model("minilm", MINILM_NPY, MINILM_DIM, minilm_rows, texts, rel, minilm_enc)

    print("\n" + "=" * 62)
    print(f"PARTIAL-INDEX A/B  (pool={P:,} docs, n={len(subset):,} queries, brute-force cosine)")
    print("NOTE: reduced pool inflates absolute scores; the MODEL DELTA is the signal.")
    print("=" * 62)
    print(f"{'Model':<16}{'MRR@10':>10}{'nDCG@10':>10}{'Recall@100':>12}")
    print("-" * 62)
    print(f"{'MiniLM (base)':<16}{m_mrr:>10.4f}{m_ndcg:>10.4f}{m_rec:>12.4f}")
    print(f"{'UniXcoder':<16}{u_mrr:>10.4f}{u_ndcg:>10.4f}{u_rec:>12.4f}")
    print(f"{'Δ (Unix-Mini)':<16}{u_mrr-m_mrr:>+10.4f}{u_ndcg-m_ndcg:>+10.4f}{u_rec-m_rec:>+12.4f}")
    print("=" * 62)
    return 0


if __name__ == "__main__":
    sys.exit(main())
