"""
CodeSearch — Gradio UI

Side-by-side comparison of four retrieval modes over CodeSearchNet Python:
    BM25 · Dense (MiniLM) · Hybrid (RRF) · Hybrid + cross-encoder rerank

Boot is NON-INDEXING by design:
    - Dense vectors live in Qdrant, built offline via scripts/index_corpus.py.
    - BM25 loads a prebuilt slim index (.cache/bm25-slim) in full-corpus mode.
    - The app refuses to embed the corpus at boot (see the guard below).

Modes:
    SMOKE_TEST_SIZE=-1  → full corpus; BM25 from the slim index. (HF Space secret)
    SMOKE_TEST_SIZE=100 → local pipeline check; BM25 built in-memory. Dense/hybrid
                          modes return full-Qdrant ids not in the small corpus, so
                          only BM25 is meaningful in smoke mode.
"""

import os
import random
import sys
import time
import traceback

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

import gradio as gr

from codesearch.config import SMOKE_TEST_SIZE, TOP_K
from codesearch.data import load_codesearch
from codesearch.retrievers.bm25 import BM25Retriever
from codesearch.retrievers.bm25_index import BM25Index
from codesearch.retrievers.dense import DenseRetriever
from codesearch.retrievers.hybrid import HybridRetriever
from codesearch.retrievers.reranker import CrossEncoderReranker

_HERE = os.path.dirname(os.path.abspath(__file__))
BM25_SLIM_DIR = os.path.join(_HERE, ".cache", "bm25-slim")

# Candidate pool for hybrid fusion and reranking in the UI. Small on purpose:
# reranking is inference-bound (~20 pairs/query on CPU), and K=20 keeps
# hybrid+rerank under the 3s latency budget. Eval uses K=100 for Recall; the UI
# trades a little recall for responsiveness. See M3.2 design notes.
UI_POOL = 20

# ---------------------------------------------------------------------------
# Boot: load corpus + wire retrievers once. No indexing happens here.
# ---------------------------------------------------------------------------

print("=== CodeSearch Boot ===")
corpus, queries = load_codesearch()  # queries = eval set (test docstrings w/ known gold)

# Two id-keyed views over the corpus:
#   corpus_by_id  — for rendering result cards (dense/hybrid hits carry id+score only)
#   corpus_lookup — reranker candidate text (code_tokens, already space-joined)
corpus_by_id = {d["id"]: d for d in corpus}
corpus_lookup = {d["id"]: d["code_tokens"] for d in corpus}

# Ground truth for the eval queries: each test docstring maps to the id of the
# function it documents. Lets the UI badge the "correct" hit and its rank — but
# only for these known queries; a free-form query has no gold.
gold_by_query = {q["query"]: q["relevant_id"] for q in queries}
eval_query_texts = list(gold_by_query.keys())

if SMOKE_TEST_SIZE == -1:
    print(f"Loading slim BM25 index from {BM25_SLIM_DIR} ...")
    bm25 = BM25Retriever(BM25Index.load_index_only(BM25_SLIM_DIR, corpus))
else:
    print(f"Smoke mode (n={SMOKE_TEST_SIZE}): building in-memory BM25 index...")
    bm25 = BM25Retriever(corpus)

dense = DenseRetriever(recreate_collection=False)
if not dense.collection_exists_and_populated():
    raise RuntimeError(
        "Qdrant collection is empty. Build it offline with scripts/index_corpus.py — "
        "the app will not embed the full corpus at boot."
    )

hybrid = HybridRetriever(bm25, dense, pool_size=UI_POOL)
reranker = CrossEncoderReranker(hybrid, corpus_lookup, pool_size=UI_POOL)

RETRIEVERS = {
    "BM25": bm25,
    "Dense (MiniLM)": dense,
    "Hybrid (RRF)": hybrid,
    "Hybrid + Rerank": reranker,
}
MODES = list(RETRIEVERS.keys())

print(f"Ready. corpus={len(corpus):,} docs · modes={MODES}")

# ---------------------------------------------------------------------------
# Search + formatting
# ---------------------------------------------------------------------------


def _fmt_latency(dt: float) -> str:
    return f"⏱ **{dt:.2f} s**" if dt >= 1.0 else f"⏱ **{dt * 1000:.0f} ms**"


def _format(hits: list[dict], mode: str, gold_id: str | None = None) -> str:
    if not hits:
        return "_No results._"

    cards = []
    for i, h in enumerate(hits, 1):
        doc = corpus_by_id.get(h["id"])
        if doc is None:
            cards.append(
                f"**#{i}** — `{h['id']}` not in local corpus "
                f"(expected in smoke mode; full corpus on the Space)."
            )
            continue

        docstring = (doc.get("docstring") or "").strip()[:200]
        code = (doc.get("code") or "").strip()[:400]
        url = doc.get("url", "")
        score = h.get("score", 0.0)

        # Provenance line: shows WHY a doc ranked where it did, per mode.
        prov = f"score **{score:.4f}**"
        if h.get("bm25_rank") is not None or h.get("dense_rank") is not None:
            prov += f" · bm25 #{h.get('bm25_rank')} · dense #{h.get('dense_rank')}"
        if "pre_rerank_score" in h:
            prov += f" · pre-rerank {h['pre_rerank_score']:.4f}"

        gold = " ✅ **GOLD**" if gold_id is not None and h["id"] == gold_id else ""
        cards.append(
            f"**#{i}**{gold} — {prov}\n\n"
            f"{docstring}{'…' if docstring else ''}\n\n"
            f"```python\n{code}{'…' if code else ''}\n```\n"
            + (f"[source]({url})\n" if url else "")
            + "\n---"
        )
    return "\n".join(cards)


def _run(mode: str, query: str, gold_id: str | None) -> tuple[str, str]:
    retr = RETRIEVERS[mode]
    t0 = time.perf_counter()
    try:
        hits = retr.retrieve(query, top_k=TOP_K)
    except Exception as e:  # one mode failing must not take down the search
        traceback.print_exc()
        return f"⚠️ **{mode} failed:** `{type(e).__name__}: {e}`", "—"
    dt = time.perf_counter() - t0

    # Gold verdict header — only for known eval queries (free-form has no gold).
    verdict = ""
    if gold_id is not None:
        rank = next((i for i, h in enumerate(hits, 1) if h["id"] == gold_id), None)
        verdict = (
            f"### ✅ Gold answer at rank #{rank}\n\n"
            if rank is not None
            else f"### ✗ Gold answer not in top-{TOP_K}\n\n"
        )
    return verdict + _format(hits, mode, gold_id), _fmt_latency(dt)


def pick_random() -> str:
    """Fill the query box with a random eval query (one whose gold we know)."""
    return random.choice(eval_query_texts) if eval_query_texts else ""


def search(query: str, left_mode: str, right_mode: str):
    """Run both columns and return (left_md, left_latency, right_md, right_latency)."""
    if not query.strip():
        msg = "_Enter a query above._"
        return msg, "", msg, ""
    # Known only for eval queries (exact match); free-form typing → gold_id None.
    gold_id = gold_by_query.get(query) or gold_by_query.get(query.strip())
    left_md, left_lat = _run(left_mode, query, gold_id)
    right_md, right_lat = _run(right_mode, query, gold_id)
    return left_md, left_lat, right_md, right_lat


# ---------------------------------------------------------------------------
# UI — selectable two-column compare
# ---------------------------------------------------------------------------

with gr.Blocks(title="CodeSearch") as demo:
    gr.Markdown(
        f"""
# 🔍 CodeSearch
Natural-language code search over **CodeSearchNet Python** ({len(corpus):,} functions).
Pick a retriever per column and compare — watch where lexical (BM25), semantic
(dense), fusion (RRF), and reranking disagree.

Hit **🎲 Random eval query** to load a real benchmark query with a known answer —
the **✅ GOLD** badge marks the correct function, and each column reports the rank it landed at.
"""
    )

    with gr.Row():
        query_box = gr.Textbox(
            placeholder="e.g. parse a JSON file and return a dict",
            label="Query",
            scale=4,
        )
        search_btn = gr.Button("Search", variant="primary", scale=1)
        random_btn = gr.Button("🎲 Random eval query", scale=1)

    with gr.Row():
        with gr.Column():
            left_mode = gr.Dropdown(MODES, value="BM25", label="Left retriever")
            left_lat = gr.Markdown()
            left_out = gr.Markdown()
        with gr.Column():
            right_mode = gr.Dropdown(
                MODES, value="Hybrid + Rerank", label="Right retriever"
            )
            right_lat = gr.Markdown()
            right_out = gr.Markdown()

    inputs = [query_box, left_mode, right_mode]
    outputs = [left_out, left_lat, right_out, right_lat]
    search_btn.click(fn=search, inputs=inputs, outputs=outputs)
    query_box.submit(fn=search, inputs=inputs, outputs=outputs)
    # Random: fill the box with a known-gold eval query, then run the comparison.
    random_btn.click(fn=pick_random, outputs=query_box).then(
        fn=search, inputs=inputs, outputs=outputs
    )


if __name__ == "__main__":
    demo.launch(theme=gr.themes.Monochrome())
