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
- Document fields used: `func_code_url` (id), `whole_func_string` (code), `func_code_tokens` (indexed by BM25), `func_documentation_string` (docstring)
- BM25 indexes **`func_code_tokens`** (pre-tokenized code, docstrings stripped by the dataset). Indexing docstrings inflates MRR to ~0.96 — verified empirically.
- Dense retrieval embeds **code**. This asymmetry is intentional and instructive.

Smoke-test mode: `SMOKE_TEST_SIZE=100` in `.env` loads 100 docs for fast iteration (M0).
Full corpus: `SMOKE_TEST_SIZE=-1`.

---

## Architecture

### Retrieval Pipeline

```
Query
  │
  ├──► BM25 (bm25s, indexes func_code_tokens)        ──► top-K BM25 hits
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
- 1 cluster, 1GB RAM
- Clusters are reaped after extended idle (~weeks); confirmed 2026-06-29 when the M2
  cluster was deleted after ~6 weeks of inactivity. Re-indexing 434k MiniLM vectors
  takes ~15-20 min on CPU, so this is recoverable but not free.
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
| BM25 | `bm25s` | Vectorized sparse BM25, ~200x faster than rank_bm25 on full corpus |
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
            ├── bm25.py           # bm25s wrapper, indexes func_code_tokens
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

### M0 · Hello World ✅
**Scope:** Touch every layer. No metrics. 100 docs only.
**Done when:** Public HF Spaces URL returns BM25 and dense results side-by-side.
**Key files:** `app.py`, `data.py`, `retrievers/bm25.py`, `retrievers/dense.py`, `config.py`

---

### M1 · BM25 Baseline + Eval Harness ✅
**Scope:** Full corpus. Build the eval loop you'll reuse forever. Establish baseline numbers.
**Files created:**
- `src/codesearch/eval/metrics.py` — MRR@10, nDCG@10, Recall@100
- `src/codesearch/eval/harness.py` — eval loop over all queries
- Updated `data.py` to full corpus loading, correct `func_code_url` IDs, `func_code_tokens` field
- Switched BM25 from `rank_bm25` to `bm25s` (vectorized, ~200x faster)

**Results (full corpus, 22k queries):** MRR@10=0.2747, nDCG@10=0.3020, Recall@100=0.5208

**Concept checkpoint:** Can explain what MRR@10=0.X means for a user, and why Recall@100
is more important than MRR@10 for evaluating the retriever stage in a RAG pipeline.

---

### M2 · Dense Retrieval ✅
**Scope:** Full corpus embedding + indexing. Dense retrieval path. Head-to-head vs BM25.

**Results (full corpus, 22k queries):** MRR@10=0.3891, nDCG@10=0.4309, Recall@100=0.7520
**Changes:**
- Add offline indexing script: `scripts/index_corpus.py` — run once, not at app boot
- Update `app.py` to skip re-indexing if collection already populated
- Add `--retriever dense` to eval harness

**Done when:** Ablation table has two rows. Can explain one query where dense wins and one
where BM25 wins with a concrete reason (not just "dense is better at semantics").

**Concept checkpoint:** Understand what HNSW `ef_search` controls. Run a sweep,
note recall and latency at each setting. See `scripts/ef_sweep.py` (reads cached
query vectors from `scripts/cache_query_vectors.py`; uses Qdrant's REST response
`time` field for server-only latency, separate from client wall time).

Sweep results (n=500 sampled queries, seed=42, top-100):

| ef  | MRR@10 | Recall@100 | server ms/q | wall ms/q |
|-----|--------|------------|-------------|-----------|
| 16  | 0.3711 | 0.7240     | 5.50        | 13.87     |
| 32  | 0.3711 | 0.7240     | 5.55        | 11.81     |
| 64  | 0.3741 | 0.7380     | 5.84        | 14.86     |
| 128 | 0.3764 | 0.7520     | 8.87        | 16.66     |
| 256 | 0.3779 | 0.7560     | 14.17       | 21.71     |
| 512 | 0.3784 | 0.7600     | 23.85       | 30.72     |

Findings:
1. **ef floor exists between 32 and 64.** ef=16 and ef=32 produce bit-identical
   metrics and server time — Qdrant clamps the effective ef to a minimum.
   Asking for ef<64 is a no-op.
2. **Recall halves with each doubling above the floor** (textbook HNSW shape):
   64→128 = +1.4pp, 128→256 = +0.4pp, 256→512 = +0.4pp. By ef=256 you've
   captured ~95% of the recall available.
3. **Server time linearizes in ef above the floor** (+1 ms per +20 ef). Below
   the floor, network RTT to Qdrant Cloud (~8 ms/query) dominates and `ef`
   tuning is invisible at the client.
4. **ef=128 is the sweet spot.** ef=64 is a viable lower-latency alternative
   (-1.4pp recall, -35% server time) if a strong reranker is downstream.
   ef=512 is never worth it here.

Recall@100 sensitivity to `ef` (3.6pp range) is much higher than MRR@10
sensitivity (0.7pp range) — the easy near-neighbors are found by any beam
width; only the deep top-100 needs a wide one. This matters for M3: the
reranker depends on Recall@100 to have anything to work with.

**Qualitative comparison.** Aggregate metrics like MRR are necessary but
insufficient — they don't show *which kinds of queries* each retriever
wins on. See `scripts/compare_retrievers.py`: for each query, compute the
rank of the relevant doc in BM25 and dense (top-100 each), then classify
by margin. "Winner" = relevant doc at rank ≤ 5; "loser" = rank > 20 or
missing. The gap between 5 and 20 is deliberate — it filters out ambiguous
cases so the remaining wins have an articulable reason.

Final bucket breakdown (full corpus, n=22,176 queries):

| Bucket | Count | % of queries |
|--------|-------|--------------|
| Dense ≫ BM25  | 4,415 | 19.9% |
| BM25 ≫ Dense  | 1,287 | 5.8% |
| Both at rank #1 | 2,765 | 12.5% |
| Gray zone (no clear winner) | 13,709 | 61.8% |

**Dense:BM25 win ratio = 3.4 : 1.** The breakdown was already within 1pp
of final at n=2,000 — the test is sample-stable.

#### Example 1 — Dense win (vocabulary gap)

**Query:**
> Calculate the SVD of a blocked RDD directly, returning only the leading
> k singular vectors. Assumes n rows and d columns, efficient when n >> d.
> Must be able to fit d^2 within the memory of a single machine.

**Ground truth** — [sparkit-learn / truncated_svd.py#L15-L52](https://github.com/lensacom/sparkit-learn/blob/0498502107c1f7dcf33cda0cdb6f5ba4b42524b7/splearn/decomposition/truncated_svd.py#L15-L52)
```python
def svd(blocked_rdd, k):
    """
    Calculate the SVD of a blocked RDD directly, returning only the leading k
    singular vectors. Assumes n rows and d columns, efficient when n >> d
    ...
```

**Dense rank #1** ✅ &nbsp; **BM25 rank >100** ❌

**What BM25 picked instead** — [apache/spark distributed.py#L1051-L1071](https://github.com/apache/spark/blob/618d6bff71073c8c93501ab7392c3cc579730f0b/python/pyspark/mllib/linalg/distributed.py#L1051-L1071)
```python
def blocks(self):
    """
    The RDD of sub-matrix blocks
    ((blockRowIndex, blockColIndex), sub-matrix) that form this
    ...
```

**Why dense wins.** `func_code_tokens` strips docstrings, so the GT's code
carries only `svd`, `blocked_rdd`, `k`, plus numpy ops — none of which are
in the query. The query is rich English ("SVD", "singular vectors",
"leading", "efficient when n >> d") that the code body simply doesn't
share. BM25 latches onto the one cross-vocabulary overlap ("blocks") and
ranks Apache Spark's `blocks()` RDD method above the actual answer.
MiniLM bridges NL→code because "SVD" and `numpy.linalg.svd`-style code
co-occur in its training data.

#### Example 2 — BM25 win (rare-identifier fingerprint)

**Query:**
> Decode value of symbol together with the extra bits.
> `>>> d = DistanceAlphabet('D', NPOSTFIX=2, NDIRECT=10)`
> `>>> d[34].value(2)`
> `(0, 35)`

**Ground truth** — [google/brotli brotlidump.py#L1145-L1165](https://github.com/google/brotli/blob/4b2b2d4f83ffeaac7708e44409fe34896a01a278/research/brotlidump.py#L1145-L1165)
```python
def value(self, dcode, dextra):
    """Decode value of symbol together with the extra bits.
    >>> d = DistanceAlphabet('D', NPOSTFIX=2, NDIRECT=10)
    >>> d[34].value(2)
    ...
```

**BM25 rank #2** ✅ &nbsp; **Dense rank >100** ❌

**What dense picked instead** — [pierre-rouanet/hampy hamming.py#L52-L82](https://github.com/pierre-rouanet/hampy/blob/bb633a3936f8a3b5f619fb0d92c7448f3dc3c92d/hampy/hamming.py#L52-L82)
```python
def decode(C):
    """ Decode data using Hamming(7, 4) code.
    ...
```

**Why BM25 wins.** The query carries three near-unique identifiers —
`DistanceAlphabet`, `NPOSTFIX`, `NDIRECT` — that are Brotli-compression-
specific and effectively unique in the corpus. BM25's IDF turns rarity
into a power tool: those terms have astronomical weight, and only docs
in `brotlidump.py` get the boost. Dense made the conceptually plausible
but wrong call: "decode + symbol + bits" maps semantically near *Hamming
code decoding*, so MiniLM surfaced a Hamming decoder from a different
library. The bi-encoder has no knowledge that "this specific codebase
uses these specific names" — rare identifiers are exactly where lexical
retrieval wins.

#### Example 3 — Tie (both signals aligned)

**Query:**
> Get historical usage metrics. webspace_name: The name of the webspace.
> website_name: The name of the website. metrics: Optional. List of
> metrics name. Otherwise, all metrics returned…

**Ground truth** — [Azure/azure-sdk-for-python websitemanagementservice.py#L210-L237](https://github.com/Azure/azure-sdk-for-python/blob/d7306fde32f60a293a7567678692bdad31e4b667/azure-servicemanagement-legacy/azure/servicemanagement/websitemanagementservice.py#L210-L237)
```python
def get_historical_usage_metrics(self, webspace_name, website_name,
                                 metrics=None, start_time=None,
                                 end_time=None, time_grain=None):
    '''
    Get historical usage metrics.
    ...
```

**BM25 rank #1** ✅ &nbsp; **Dense rank #1** ✅

**Why both win.** The function name and parameter names —
`get_historical_usage_metrics`, `webspace_name`, `website_name`,
`metrics` — are paraphrases of the query itself. BM25 has direct token
overlap on rare identifiers (high IDF). Dense has direct semantic match
because the function is named after what it does. When the query is
essentially a natural-language reading of an already-descriptive Python
signature, neither retriever is the bottleneck — they just agree.

#### Twin-function disambiguation (M3 motivation)

Two final examples illustrate the *same phenomenon from opposite
directions* — neither retriever reliably disambiguates near-twin
functions in the same file.

**Twin example A — dense disambiguates a single-token swap.**

Query:
> Downloads a Sina video by its unique **vid**.
> `http://video.sina.com.cn/`

Ground truth — [soimort/you-get sina.py#L41-L52](https://github.com/soimort/you-get/blob/b746ac01c9f39de94cac2d56f665285b0523b974/src/you_get/extractors/sina.py#L41-L52)
```python
def sina_download_by_vid(vid, title=None, output_dir='.',
                         merge=True, info_only=False):
    """Downloads a Sina video by its unique vid.
    http://video.sina.com.cn/
    """
```

**Dense rank #1** ✅ &nbsp; **BM25 rank >100** ❌

BM25's top-1 was the *near-twin* in the same file — [sina.py#L54-L64](https://github.com/soimort/you-get/blob/b746ac01c9f39de94cac2d56f665285b0523b974/src/you_get/extractors/sina.py#L54-L64):
```python
def sina_download_by_vkey(vkey, title=None, output_dir='.',
                          merge=True, info_only=False):
    """Downloads a Sina video by its unique vkey.
    http://video.sina.com/
    """
```

With docstrings stripped, the two functions' `func_code_tokens` are
near-identical to BM25 (`def`, `sina_download_by_v(id|key)`, parameter
names…). The single disambiguating token in the query ("vid") gets
diluted in sparse scoring. Dense weighs it correctly because in the
embedding space `vid` carries the "video-ID concept" signal distinctly
from `vkey`.

**Twin example B — BM25 confuses a verb swap.**

Query:
> Post the content of a URL via sending a HTTP **POST** request.
> Args: url: A URL. headers: Request headers used by the client.
> decoded: Whether decode the response body…

Ground truth — [soimort/you-get common.py#L457-L501](https://github.com/soimort/you-get/blob/b746ac01c9f39de94cac2d56f665285b0523b974/src/you_get/common.py#L457-L501)
```python
def post_content(url, headers={}, post_data={}, decoded=True, **kwargs):
    """Post the content of a URL via sending a HTTP POST request.
    Args:
    ...
```

**BM25 rank #3** ⚠️ &nbsp; **Dense rank >100** ❌

BM25 put the GT in top-5 but its top-1 was the *GET* twin — [common.py#L415-L454](https://github.com/soimort/you-get/blob/b746ac01c9f39de94cac2d56f665285b0523b974/src/you_get/common.py#L415-L454):
```python
def get_content(url, headers={}, decoded=True):
    """Gets the content of a URL via sending a HTTP GET request.
    ...
```

BM25's lexical signal couldn't distinguish "get" vs "post" reliably enough
— both functions live in the same file, share parameter names, and have
docstrings stripped. Dense missed both versions and generalized out to a
`post()` method in a boatd HTTP client library.

**The symmetry is the M3 pitch.** Each retriever fails on different
near-twins, but **neither succeeds at twin-disambiguation reliably**.
RRF alone won't fix this (both rank lists prefer the wrong twin in
example B). A cross-encoder, which scores query-and-candidate-code
*as a pair*, is the decisive piece — it can read the actual verb
("Post" vs "Gets") in both query and code body simultaneously.

#### Gray-zone significance

61.8% of queries fall into the no-clear-winner zone (both retrievers in
rank 6–100, or both missing entirely). This is the biggest opportunity
surface for M3:

- **RRF** promotes "rank 15 in both lists" candidates into a fused top-5
  via `1/(60+15) + 1/(60+15)` ≈ 0.027 — beating singletons that only
  appear in one list.
- **Cross-encoder reranker** then lifts the right candidate to #1 by
  reading query and code side-by-side.

#### Deferred from M2 — see auto-memory TODOs

- Single-query latency benchmark (P50/P90/P95 BM25 vs dense end-to-end)
- Cost comparison axis (storage, build/serve compute, $/mo at target QPS)

---

### M3 · Hybrid Retrieval + Reranking ✅
**Scope:** RRF fusion. Cross-encoder reranker. Full ablation table.

**Results:**

| Retriever          | MRR@10   | nDCG@10  | Recall@100 | Sample     |
|--------------------|----------|----------|------------|------------|
| Hybrid RRF (M3.1)  | 0.3977   | 0.4475   | 0.7750     | full 22k   |
| Hybrid + rerank (M3.2) | 0.4011   | 0.4486   | 0.7750     | full 22k   |

**M3.1 finding — RRF's signature lift is on Recall, not MRR.** +2.3pp Recall@100 vs dense, +0.9pp MRR@10. Fusion widens the top-100 pool for the reranker to pick from; rank-1 lift was expected to be M3.2's job.

**M3.2 finding — off-the-shelf MS-MARCO CE gives only a marginal lift.** Full 22k eval: +0.34pp MRR (0.3977→0.4011), +0.11pp nDCG (0.4475→0.4486) vs hybrid alone; Recall unchanged by construction (reranker only reorders). The full eval confirms a real but tiny positive effect and resolves the earlier n=2000 confusion (where the sampled rerank MRR, 0.3938, sat slightly below full-hybrid). The ~24h CPU cost per full run (2.2M CE forward passes at ~26 pair/s) is a poor trade for +0.3pp. The most likely explanation is the domain gap: MS-MARCO CE is an NL passage ranker being asked to score NL-query / space-joined-AST-token pairs. Code-aware CE (CodeBERT / UniXcoder) and/or richer candidate text (AST-stripped `whole_func_string`) are the M5 upside.

**RRF implementation note:** `score(d) = 1/(60 + rank_bm25(d)) + 1/(60 + rank_dense(d))`
Documents that only appear in one list get `1/(60 + K+1)` for the missing list (treat as rank K+1).
k=60 is standard; don't tune it.

**Concept checkpoint:** Articulated why RRF is rank-based rather than score-based (BM25 log-IDF scores and cosine similarities live on incompatible scales; RRF sidesteps the normalisation problem by voting on ranks), and what a cross-encoder attends to that a bi-encoder cannot (joint attention over query+doc tokens, so query terms can gate on doc structure — but only if the CE was trained on comparable pair distributions, which MS-MARCO wasn't for code).

---

### M4 · UI + Full Deployment
**Scope:** Polish Gradio UI. Full corpus on HF Spaces. Public URL in README.
**Changes:**
- Gradio UI: retriever mode selector, latency display, cleaner result cards
- HF Spaces: set `SMOKE_TEST_SIZE=-1` in secrets, verify indexing is offline (not at boot)
- README: add public URL, final ablation table

**Done when:** Public URL. All 4 retrieval modes work. End-to-end latency < 3s for hybrid+rerank.

---

### M5 · Embedding Experiments (in progress)
**Scope:** Is the dense bottleneck the *representation* or the *model*? Two sub-experiments.

**Sub-experiment A — doc representation (DONE, null).** Embed docstring-stripped function
source (`data.strip_docstring`) vs CSN `code_tokens`, MiniLM fixed. Same-sample A/B
(n=2,000, seed=42): Dense MRR 0.3938→0.3938 (±0.00), Hybrid 0.3875→0.3853 — all inside
SE≈0.01. **A general NL embedder can't exploit code structure, so representation barely
moves MRR — the model is the bottleneck.** (Full-22k stripped eval failed: on-disk Qdrant
queries time out at 22k batch scale; n=2,000 A/B is the evidence. To finish the full row,
bump the DenseRetriever Qdrant client timeout, or cite n=2,000.)

**Sub-experiment B — swap to a fair code bi-encoder (indexing + eval underway).**
Model: **`microsoft/unixcoder-base`** — code-pretrained, *not* CSN-retrieval-finetuned
(a fair test; CSN-tuned encoders like `st-codesearch-distilroberta-base` are rigged
in-distribution and kept only as a labelled ceiling). Jina's `jina-embeddings-v2-base-code`
is dead on our stack (`find_pruneable_heads_and_indices` removed in transformers 5.x).
- **Faithful encoder:** the official `unixcoder.py` can't run under transformers 5.x (its
  2-D attention mask crashes batched / goes causal single). `codesearch/embedding.py`
  reproduces the official encoder-only recipe (mode token → mean-pool → L2-norm) in a
  5.x-executable form, validated **element-wise identical** (`max|Δ|=0`) to the official
  model run bidirectionally (`is_decoder` proven mask-only, zero weight effect). Test:
  `scripts/test_unixcoder_encoder.py`.
- **Fair setup:** `EMBED_MAX_LENGTH=256` to match MiniLM's `max_seq_length` (apples-to-apples).
  768-dim × 434k > 1GB free tier → `QDRANT_ON_DISK=1`. ~37h CPU index (already core-bound;
  no fair speedup — parallelism ~1.4× max, quantization would confound).
- **Early read:** `GOLDS_FIRST=1` embeds the ~22k eval-gold docs first, so
  `scripts/sample_eval_partial.py` runs a fair brute-force cosine A/B (both models, same pool,
  exact) straight from the `.npy` caches at partial milestones before the full index finishes.

**Done when:** README ablation table UniXcoder row filled. Can answer:
"Does model choice or architecture choice (hybrid+rerank) give a bigger MRR lift?"

---

## Key Constraints & Gotchas

1. **Never embed at app boot for the full corpus.** Embedding 400k docs takes minutes.
   Use an offline `scripts/index_corpus.py` run once locally or as a one-time HF Space task.
   `dense.py` already checks `collection_exists_and_populated()` and skips if true.

2. **Qdrant free tier: 1GB RAM.** 384-dim in-RAM (MiniLM) fits. Larger models (e.g.
   768-dim UniXcoder ≈ 1.3GB) exceed it — set `QDRANT_ON_DISK=1` (config `QDRANT_ON_DISK`,
   passed to `VectorParams(on_disk=True)`): vectors live on disk, HNSW graph stays in RAM.
   Fine for a batch eval. Alternatively reduce corpus slice or upgrade tier.

3. **BM25 scores and cosine similarities cannot be directly compared or linearly combined.**
   They are on completely different scales. Always use RRF for fusion.

4. **HF Spaces has no GPU.** Query-time embedding (one vector) is fast on CPU (~50ms).
   Reranking 100 pairs is slower (~500ms-1s on CPU). This is acceptable for a demo.

5. **CodeSearchNet has one relevant doc per query.** nDCG with binary single-relevant
   judgments behaves like reciprocal rank. Both MRR and nDCG will tell similar stories here.
   This is fine for learning; real-world eval datasets have graded multi-relevant judgments.
