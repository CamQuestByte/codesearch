---
title: Codesearch
emoji: 🔍
colorFrom: purple
colorTo: purple
sdk: gradio
sdk_version: 6.13.0
app_file: app.py
pinned: false
license: mit
short_description: BM25 vs dense retrieval on CodeSearchNet
---

# CodeSearch: Semantic Retrieval from Scratch

Learning project: BM25 vs dense vs hybrid retrieval on CodeSearchNet, with rigorous eval.

**🔍 Live demo:** https://camquest-codesearch.hf.space — pick a retriever per column and compare BM25 / dense / hybrid / reranked results side-by-side, with per-query latency.

## Retrieval Results

### Reference numbers (Husain et al., 2019)

Eval setup: 1+999 random distractors per query (not full corpus). Only MRR is reported. BM25 indexes docstrings — inflated relative to a proper code-search baseline.

| Retriever | MRR | Notes |
|-----------|-----|-------|
| BM25 (Elasticsearch) | ~0.68 | docstring index |

### Our progress

Eval setup: full 434k-doc corpus, 22k queries, BM25 indexes `func_code_tokens` (docstrings stripped).

| Retriever | Model | MRR@10 | nDCG@10 | Recall@100 | Milestone |
|-----------|-------|--------|---------|------------|-----------|
| BM25 | — | 0.2747 | 0.3020 | 0.5208 | M1 ✅ |
| Dense | MiniLM-L6-v2 | 0.3891 (+0.1144) | 0.4309 (+0.1289) | 0.7520 (+0.2312) | M2 ✅ |
| Hybrid (RRF) | MiniLM + BM25 | 0.3977 (+0.1230) | 0.4475 (+0.1455) | 0.7750 (+0.2542) | M3.1 ✅ |
| Hybrid + Rerank | + `ms-marco-MiniLM-L-6-v2` | 0.4011 (+0.1264) | 0.4486 (+0.1466) | 0.7750 (+0.2542) | M3.2 ✅ |
| Dense | UniXcoder-base _(fair code model)_ | 0.4343 (+0.1596) | 0.4748 (+0.1728) | 0.7769 (+0.2561) | M5 ✅ |

Deltas are vs BM25 baseline. **Recall@100 jumps 23.1pp** going from sparse to dense — the model bridges the natural-language-query → code vocabulary gap that BM25 cannot. This is the motivating data point for M3: there is 23pp of headroom available to a reranker on top of dense, but only ~8pp on top of BM25 alone.

**M3.1 — Hybrid RRF lifts Recall@100 by +2.3pp on top of dense** (0.7520 → 0.7750), with a much smaller MRR@10 lift (+0.86pp). That's the expected RRF signature: fusion promotes "rank-15 in both lists" candidates into the fused top-50 (fattening the pool for a downstream reranker), but it rarely lifts anything to rank-1 by itself. The Recall headroom from set-union over the same K is +5pp above RRF — flagged as the lever to pull if M3.2 underperforms.

**M3.2 — the MS MARCO cross-encoder produces only a marginal lift over hybrid RRF.** Full 22k eval: hybrid MRR@10=0.3977 → hybrid+rerank 0.4011 (+0.34pp); nDCG@10 0.4475 → 0.4486 (+0.11pp); Recall@100 identical (0.7750) by construction — the reranker only reorders the pool it's given. The full eval confirms a real but tiny positive effect and resolves an earlier n=2000 sampling artifact (where the sampled rerank MRR, 0.3938, sat slightly *below* full-hybrid, SE≈0.01). Cost: ~24h CPU per full run (2.2M CE forward passes at ~26 pair/s on the eval laptop) — poor ROI for +0.3pp. The most likely explanation is the domain gap: MS MARCO CE was trained on NL→NL passage ranking and reads `code_tokens` (space-joined AST tokens) as an alien input. **The honest takeaway: off-the-shelf NL cross-encoders give only a marginal (~+0.3pp) improvement to hybrid RRF on NL→code retrieval — real, but not worth the compute.** M5 explores whether code-aware CEs (CodeBERT / UniXcoder-based) or richer candidate text (AST-stripped `whole_func_string`) can produce a real lift.

### M5 — embedding experiments (in progress)

**Sub-experiment A — does the doc *representation* matter? No (null result).** Re-embedded the corpus with the docstring-stripped **function source** (`data.strip_docstring` — AST span-excision, validated 0 docstring leakage) instead of CSN `code_tokens`, holding the model (MiniLM-L6-v2) fixed. Same-sample A/B (n=2,000, seed=42): Dense MRR@10 0.3938 → 0.3938 (±0.00), Hybrid 0.3875 → 0.3853 — every delta inside SE≈0.01. Interpretation: a general-purpose NL embedder can't exploit code structure, so *how* the code is serialised barely moves retrieval. The **model** is the bottleneck, not the representation — which is what motivates B.

**Sub-experiment B — swap in a *fair* code bi-encoder: +4.5pp MRR, the biggest lever in the project.** Replacing MiniLM with **UniXcoder-base** (code-pretrained but *not* CodeSearchNet-retrieval-finetuned — an honest test; CSN-tuned encoders would be rigged in-distribution) lifts dense **MRR@10 0.3891 → 0.4343 (+4.5pp)**, nDCG@10 0.4309 → 0.4748, Recall@100 0.7520 → 0.7769 — full 434k corpus, 22k queries, same `code_tokens` input, same Qdrant HNSW method. For scale that's ~5× the hybrid-RRF gain (+0.9pp) and ~14× the cross-encoder rerank gain (+0.3pp). **This confirms the A→B thesis: the *model* was the dense bottleneck, not the representation.**

Implementation notes: the official `unixcoder.py` can't run on our pinned `transformers 5.x` — its 2-D `(mask_i·mask_j)` attention mask crashes on batched input and goes silently causal on single input — so `codesearch/embedding.py` reproduces the official encoder-only recipe (mode-token framing → mean-pool → L2-normalize) in a form 5.x executes, validated **element-wise identical** (`max|Δ|=0`) to the official model run bidirectionally. Held to the same `max_seq_length=256` as the MiniLM baseline (apples-to-apples *model* swap). 768-dim × 434k > 1GB free tier → the Qdrant collection is `on_disk` (≈0.97s/query vs MiniLM's in-RAM ms; a raised client timeout + query-retry guard handle the slow reads). Next: a **code-to-code** variant (UniXcoder on AST-docstring-stripped `whole_func_string`, its native input) and a UniXcoder **hybrid** row.

## Stack

- **Dataset:** CodeSearchNet Python split (~400k functions, 4k eval queries)
- **BM25:** `bm25s` (vectorized; ~200x faster than `rank_bm25` on full corpus)
- **Embeddings:** `sentence-transformers` (local, no API cost)
- **Vector DB:** Qdrant Cloud (free tier, 1GB)
- **UI:** Gradio on Hugging Face Spaces (free)

## Milestones

| Milestone | Status |
|-----------|--------|
| M0 · Hello World | ✅ done |
| M1 · BM25 Baseline + Eval | ✅ done |
| M2 · Dense Retrieval | ✅ done |
| M3 · Hybrid + Reranking | ✅ done |
| M4 · UI + Deployment | ✅ done — [live](https://camquest-codesearch.hf.space) |
| M5 · Model Swap (optional) | ⬜ not started |

## Local Setup

```bash
# Install uv if you don't have it
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create env and install deps
uv venv
source .venv/bin/activate
uv sync

# Copy env template and fill in your Qdrant credentials
cp .env.example .env
# edit .env

# Run locally
python app.py
```

## HF Spaces Deployment

1. Create a new Space (Gradio SDK) at huggingface.co/spaces
2. Add `QDRANT_URL` and `QDRANT_API_KEY` as Space secrets
3. Push this repo: `git push hf main`

The `requirements.txt` (generated by `uv export`) is what HF Spaces uses to install deps.
To regenerate it after adding packages: `uv export --no-hashes > requirements.txt`
