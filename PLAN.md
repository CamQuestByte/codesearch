# CodeSearch — Project Plan

> This file is the source of truth for architecture decisions, stack choices, and milestone scope.
> It exists so Claude Code has full context when helping with implementation.
> Update it when decisions change. Do not let it drift from reality.

---

## Project Goal

Build a semantic code search system over CodeSearchNet (Python split) that demonstrates
when dense retrieval beats BM25, when it loses, and how hybrid approaches combine both.
Evaluate rigorously using MRR@10, nDCG@10, and Recall@100.

This is a learning project with a shippable output: a public Gradio UI on HF Spaces.
The owner has deep Lucene/BM25 experience and is ramping up on modern LLM retrieval tooling.

---

## Dataset

**CodeSearchNet Python split** (via HuggingFace `datasets` library, `code_search_net`).

- ~400k Python functions as corpus documents
- ~4k eval queries (natural language descriptions) with ground-truth relevant doc per query
- One relevant document per query (binary relevance judgments)
- Document fields used: `func_name` (id), `whole_func_string` (code), `func_documentation_string` (docstring), `func_code_url` (url)
- BM25 indexes **docstrings**. Dense retrieval embeds **code**. This asymmetry is intentional and instructive.

Smoke-test mode: `SMOKE_TEST_SIZE=100` in `.env` loads 100 docs for fast iteration (M0).
Full corpus: `SMOKE_TEST_SIZE=-1`.

---

## Architecture

### Retrieval Pipeline

```
Query
  │
  ├──► BM25 (rank_bm25, indexes docstrings)          ──► top-K BM25 hits
  │
  ├──► Dense encoder → query vector → Qdrant HNSW    ──► top-K dense hits
  │
  └──► Reciprocal Rank Fusion: score = Σ 1/(60 + rank)
           │
           └──► merged top-N candidates
                    │
                    └──► Cross-encoder reranker ──► final top-k
```

RRF is rank-based (not score-based) so BM25 scores and cosine similarities are never
directly combined — they are incommensurable and normalization would be fragile.

### Embedding Model

Default: `sentence-transformers/all-MiniLM-L6-v2` (384 dims, fast on CPU, no API cost).
M5 experiment: swap to a code-specific model (e.g. `flax-sentence-embeddings/st-codesearch-distilroberta-base`).
Model is controlled by `EMBEDDING_MODEL` env var. Changing the model requires re-indexing.

Embeddings are computed **locally** with sentence-transformers. No OpenAI API dependency.

### Vector DB

**Qdrant Cloud free tier.**
- 1 cluster, 1GB RAM, no expiry
- ~400k × 384-dim vectors ≈ 600MB — fits with room to spare
- Do not use 768-dim models or you will be tight on free tier
- Collection name controlled by `QDRANT_COLLECTION` env var
- Supports sparse + dense in one collection (needed for M3 hybrid)

### Reranker (M3)

`cross-encoder/ms-marco-MiniLM-L-6-v2` via sentence-transformers `CrossEncoder`.
Reranks top-100 candidates from hybrid retrieval down to top-10.
Note the latency tax: reranking 100 pairs is ~10x slower than a single vector search.
In production you'd rerank top-20; keep at top-100 for eval accuracy in this project.

### UI

Gradio. Single `app.py` entrypoint.
Retriever mode selector: BM25 / Dense / Hybrid / Hybrid+Rerank.
Shows: code snippet, docstring, score, latency per query.

### Deployment

**Hugging Face Spaces** (Gradio SDK, free tier: 2 vCPU, 16GB RAM, no GPU, no sleep-on-idle).
HF Spaces uses `requirements.txt` (not `pyproject.toml`) — regenerate with:
```bash
uv export --no-hashes > requirements.txt
```
Secrets (`QDRANT_URL`, `QDRANT_API_KEY`) set in Space settings, not committed to repo.

---

## Stack Decisions

| Layer | Choice | Rationale |
|-------|--------|-----------|
| Python | 3.11+ | |
| Package manager | `uv` | Speed, lockfile, `uv add` workflow |
| BM25 | `rank_bm25` | Lightweight, no server |
| Embeddings | `sentence-transformers` | Local, swappable, no API cost |
| Vector DB | Qdrant Cloud | Free tier, no expiry, sparse+dense support |
| Reranker | `sentence-transformers CrossEncoder` | Same library, no new dependency |
| UI | Gradio | Native on HF Spaces |
| Deployment | HF Spaces | Free, no sleep, Gradio-native |
| Eval metrics | `ranx` or manual numpy | MRR@10, nDCG@10, Recall@100 |

---

## Project Structure

```
codesearch/
├── app.py                        # Gradio entrypoint — run this
├── pyproject.toml
├── requirements.txt              # for HF Spaces (uv export)
├── .env.example                  # template — copy to .env, never commit .env
├── .gitignore
├── README.md                     # public-facing, updated each milestone
├── PLAN.md                       # this file
└── src/
    └── codesearch/
        ├── __init__.py
        ├── config.py             # all env vars loaded here, nowhere else
        ├── data.py               # CodeSearchNet loader, smoke-test mode
        └── retrievers/
            ├── __init__.py
            ├── bm25.py           # rank_bm25 wrapper
            ├── dense.py          # sentence-transformers + Qdrant
            ├── hybrid.py         # RRF fusion (M3)
            └── reranker.py       # CrossEncoder reranker (M3)
        └── eval/                 # added in M1
            ├── __init__.py
            ├── metrics.py        # MRR@10, nDCG@10, Recall@100
            └── harness.py        # eval loop over all queries
```

Files not yet created are listed here so Claude Code knows what to build next.

---

## Eval Metrics

All metrics computed over the full ~4k eval query set (after M1).

| Metric | Definition | Why it matters here |
|--------|-----------|-------------------|
| MRR@10 | Mean of 1/rank of first relevant hit, capped at 10 | CodeSearchNet canonical metric |
| nDCG@10 | Discounted cumulative gain at 10 (binary relevance here ≈ MAP) | Standard BEIR metric |
| Recall@100 | % of queries where relevant doc appears in top-100 | Retriever health; what a reranker needs to work with |

Recall@100 is the primary **retriever** metric. If it's low, no reranker can save you.
MRR@10 is the primary **end-to-end** metric.

Eval entry point (M1+):
```bash
python -m codesearch.eval.harness --retriever [bm25|dense|hybrid|hybrid_rerank]
```

---

## Milestones

### M0 · Hello World ✅ / 🔄
**Scope:** Touch every layer. No metrics. 100 docs only.
**Done when:** Public HF Spaces URL returns BM25 and dense results side-by-side.
**Key files:** `app.py`, `data.py`, `retrievers/bm25.py`, `retrievers/dense.py`, `config.py`

---

### M1 · BM25 Baseline + Eval Harness
**Scope:** Full corpus. Build the eval loop you'll reuse forever. Establish baseline numbers.
**New files to create:**
- `src/codesearch/eval/metrics.py` — implement MRR@10, nDCG@10, Recall@100
- `src/codesearch/eval/harness.py` — iterate eval queries, call retriever, aggregate metrics
- Update `data.py` to support full corpus loading (`SMOKE_TEST_SIZE=-1`)

**Done when:** `python -m codesearch.eval.harness --retriever bm25` prints a metrics table.
Baseline numbers committed to README ablation table.

**Concept checkpoint:** Can explain what MRR@10=0.X means for a user, and why Recall@100
is more important than MRR@10 for evaluating the retriever stage in a RAG pipeline.

---

### M2 · Dense Retrieval
**Scope:** Full corpus embedding + indexing. Dense retrieval path. Head-to-head vs BM25.
**Changes:**
- Add offline indexing script: `scripts/index_corpus.py` — run once, not at app boot
- Update `app.py` to skip re-indexing if collection already populated
- Add `--retriever dense` to eval harness

**Done when:** Ablation table has two rows. Can explain one query where dense wins and one
where BM25 wins with a concrete reason (not just "dense is better at semantics").

**Concept checkpoint:** Understand what HNSW `ef` parameter controls. Run one experiment:
`ef=32` vs `ef=128` vs `ef=512` — note recall and latency at each setting.

---

### M3 · Hybrid Retrieval + Reranking
**Scope:** RRF fusion. Cross-encoder reranker. Full ablation table.
**New files to create:**
- `src/codesearch/retrievers/hybrid.py` — RRF over BM25 + dense ranked lists
- `src/codesearch/retrievers/reranker.py` — CrossEncoder wrapper
- Add `--retriever hybrid` and `--retriever hybrid_rerank` to eval harness

**RRF implementation note:** `score(d) = 1/(60 + rank_bm25(d)) + 1/(60 + rank_dense(d))`
Documents that only appear in one list get `1/(60 + K+1)` for the missing list (treat as rank K+1).
k=60 is standard; don't tune it.

**Done when:** Four-row ablation table (BM25, Dense, Hybrid, Hybrid+Rerank) with
MRR@10, nDCG@10, Recall@100, and latency. Each delta is explained in README notes.

**Concept checkpoint:** Articulate why RRF is rank-based rather than score-based,
and what a cross-encoder attends to that a bi-encoder cannot.

---

### M4 · UI + Full Deployment
**Scope:** Polish Gradio UI. Full corpus on HF Spaces. Public URL in README.
**Changes:**
- Gradio UI: retriever mode selector, latency display, cleaner result cards
- HF Spaces: set `SMOKE_TEST_SIZE=-1` in secrets, verify indexing is offline (not at boot)
- README: add public URL, final ablation table

**Done when:** Public URL. All 4 retrieval modes work. End-to-end latency < 3s for hybrid+rerank.

---

### M5 · Embedding Model Swap (Optional)
**Scope:** Swap `all-MiniLM-L6-v2` for a code-specific model. Measure delta.
**Candidate models:**
- `flax-sentence-embeddings/st-codesearch-distilroberta-base`
- `microsoft/codebert-base` (requires custom pooling — more work)

**Done when:** README ablation table has a fifth row. Can answer:
"Does model choice or architecture choice (hybrid+rerank) give a bigger MRR lift?"

---

## Key Constraints & Gotchas

1. **Never embed at app boot for the full corpus.** Embedding 400k docs takes minutes.
   Use an offline `scripts/index_corpus.py` run once locally or as a one-time HF Space task.
   `dense.py` already checks `collection_exists_and_populated()` and skips if true.

2. **Qdrant free tier: 1GB RAM.** Stick to 384-dim models. If you hit memory errors,
   reduce to a smaller corpus slice or upgrade tier.

3. **BM25 scores and cosine similarities cannot be directly compared or linearly combined.**
   They are on completely different scales. Always use RRF for fusion.

4. **HF Spaces has no GPU.** Query-time embedding (one vector) is fast on CPU (~50ms).
   Reranking 100 pairs is slower (~500ms-1s on CPU). This is acceptable for a demo.

5. **CodeSearchNet has one relevant doc per query.** nDCG with binary single-relevant
   judgments behaves like reciprocal rank. Both MRR and nDCG will tell similar stories here.
   This is fine for learning; real-world eval datasets have graded multi-relevant judgments.
