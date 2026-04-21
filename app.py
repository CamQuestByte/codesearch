"""
M0 Hello World — Gradio UI

Shows BM25 and dense results side-by-side for the same query.
Uses SMOKE_TEST_SIZE docs (default: 100).

To run locally:
    python app.py

To deploy to HF Spaces:
    - Push this repo to a HF Space (Gradio SDK)
    - Set QDRANT_URL, QDRANT_API_KEY as Space secrets
    - Set SMOKE_TEST_SIZE=100 for M0, -1 for full corpus (M4)
"""

import gradio as gr

from codesearch.data import load_codesearch
from codesearch.retrievers.bm25 import BM25Retriever
from codesearch.retrievers.dense import DenseRetriever
from codesearch.config import TOP_K

# ---------------------------------------------------------------------------
# Boot: load data + build indexes once at startup
# ---------------------------------------------------------------------------

print("=== CodeSearch M0 Boot ===")
corpus, queries = load_codesearch()

print("Building BM25 index...")
bm25 = BM25Retriever(corpus)

print("Initializing dense retriever...")
dense = DenseRetriever(recreate_collection=False)
dense.index_corpus(corpus)

print("Ready.")

# ---------------------------------------------------------------------------
# Search function
# ---------------------------------------------------------------------------

def search(query: str) -> tuple[str, str]:
    """Run BM25 and dense retrieval, return formatted results for both columns."""
    if not query.strip():
        return "Enter a query above.", "Enter a query above."

    bm25_results = bm25.retrieve(query, top_k=TOP_K)
    dense_results = dense.retrieve(query, top_k=TOP_K)

    return _format_results(bm25_results), _format_results(dense_results)


def _format_results(results: list[dict]) -> str:
    if not results:
        return "No results."

    lines = []
    for i, r in enumerate(results, 1):
        score = r.get("score", 0.0)
        docstring = (r["docstring"] or "").strip()[:200]
        code_preview = (r["code"] or "").strip()[:300]
        url = r.get("url", "")

        lines.append(
            f"**#{i} — score: {score:.4f}**\n"
            f"**Docstring:** {docstring}{'…' if len(r.get('docstring','')) > 200 else ''}\n\n"
            f"```python\n{code_preview}{'…' if len(r.get('code','')) > 300 else ''}\n```\n"
            + (f"[source]({url})\n" if url else "")
            + "\n---"
        )

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------

with gr.Blocks(title="CodeSearch M0", theme=gr.themes.Monochrome()) as demo:
    gr.Markdown(
        """
# 🔍 CodeSearch — M0 Hello World
**Dataset:** CodeSearchNet Python ({n} docs)  
**Retrieval modes:** BM25 (left) vs Dense / HNSW (right)  
Notice where they agree and where they diverge — that gap is what this project is about.
""".format(n=len(corpus))
    )

    with gr.Row():
        query_box = gr.Textbox(
            placeholder="e.g. parse a JSON file and return a dict",
            label="Query",
            scale=4,
        )
        search_btn = gr.Button("Search", variant="primary", scale=1)

    with gr.Row():
        bm25_out = gr.Markdown(label="BM25 results")
        dense_out = gr.Markdown(label="Dense (MiniLM) results")

    with gr.Row():
        gr.Markdown(
            "_Tip: try a query like 'convert string to datetime' — "
            "BM25 needs those exact words, dense finds semantically similar code._"
        )

    search_btn.click(fn=search, inputs=query_box, outputs=[bm25_out, dense_out])
    query_box.submit(fn=search, inputs=query_box, outputs=[bm25_out, dense_out])


if __name__ == "__main__":
    demo.launch()
