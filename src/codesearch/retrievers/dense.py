"""
Dense retriever: sentence-transformers bi-encoder + Qdrant HNSW index.

Lucene analogy:
    - DenseRetriever.index_corpus()  ≈  IndexWriter.addDocuments()
    - DenseRetriever.retrieve()      ≈  IndexSearcher.search() with a KnnFloatVectorQuery
    - The embedding model            ≈  Analyzer, but maps text → float[384] not tokens
    - Qdrant HNSW graph              ≈  Lucene's HNSW (yes, Lucene has one since 9.x)

Key difference from BM25:
    - BM25 indexes docstrings, queries hit docstrings (symmetric)
    - Dense indexes CODE via embeddings, queries are natural language (asymmetric)
    - This is intentional: embeddings learn a shared semantic space across the gap
    - It's also why dense beats BM25 on CodeSearchNet: the model bridges natural
      language queries to code that doesn't share any vocabulary with the query

HNSW `ef_search` parameter (set at query time):
    - Controls beam width during graph traversal
    - Higher ef_search = more accurate (higher recall) but slower
    - Default 128 is a good starting point; we'll experiment in M2
    - Equivalent intuition: Lucene's maxClauseCount isn't a perfect analogy,
      but the recall/latency tradeoff dial is the same concept
"""

from __future__ import annotations

import pickle
import random
import time
from pathlib import Path

import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct,
    SearchRequest,
    SearchParams,
)
from qdrant_client.http.models import QueryRequest
from tqdm import tqdm

from codesearch.config import (
    QDRANT_URL,
    QDRANT_API_KEY,
    QDRANT_COLLECTION,
    EMBEDDING_MODEL,
    EMBEDDING_DIM,
)
from codesearch.embedding import load_encoder

# Batch size for embedding. MiniLM is small; 256 is fast on CPU.
# Reduce to 64 if you hit memory issues on HF Spaces.
_BATCH_SIZE = 256

_QUERY_CACHE_DIR = Path(".cache")


def _query_cache_path(model_name: str) -> Path:
    safe = model_name.replace("/", "_")
    return _QUERY_CACHE_DIR / f"query_vectors_{safe}.pkl"


class DenseRetriever:
    def __init__(self, recreate_collection: bool = False) -> None:
        """
        Initialize the encoder and Qdrant client.
        Does NOT index anything — call index_corpus() explicitly.

        Args:
            recreate_collection: if True, drops and recreates the Qdrant collection.
                                 Use this when re-indexing with a new model.
        """
        print(f"Loading embedding model: {EMBEDDING_MODEL}")
        self.model = load_encoder(EMBEDDING_MODEL)

        print(f"Connecting to Qdrant at {QDRANT_URL}")
        # timeout raised well above the 5s default: on-disk collections (768-dim
        # models that exceed the 1GB free-tier RAM) read vectors from disk per
        # query, so a search batch can take tens of seconds. In-RAM collections
        # (MiniLM) never need this.
        self.client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY, timeout=120)
        self.collection = QDRANT_COLLECTION

        self._query_cache: dict[str, np.ndarray] = {}
        self._try_load_query_cache()

        if recreate_collection:
            self._create_collection()

    def _try_load_query_cache(self) -> None:
        """Populate self._query_cache from .cache/query_vectors_<model>.pkl if present.

        The cache is built once by scripts/cache_query_vectors.py and keyed by
        query text → np.ndarray. Silently no-ops if the file is missing; warns
        and ignores if the file was built for a different embedding model.
        """
        path = _query_cache_path(EMBEDDING_MODEL)
        if not path.exists():
            return
        with open(path, "rb") as f:
            data = pickle.load(f)
        if data.get("model") != EMBEDDING_MODEL:
            print(
                f"[dense] query-vector cache at {path} was built for "
                f"{data.get('model')!r}, current model is {EMBEDDING_MODEL!r} — ignoring."
            )
            return
        vectors = data["vectors"]
        self._query_cache = {
            q["query"]: vectors[i] for i, q in enumerate(data["queries"])
        }
        print(
            f"[dense] loaded {len(self._query_cache):,} cached query vectors from {path}"
        )

    def _create_collection(self) -> None:
        """Create (or recreate) the Qdrant collection."""
        # Delete if exists
        existing = [c.name for c in self.client.get_collections().collections]
        if self.collection in existing:
            print(f"Deleting existing collection '{self.collection}'...")
            self.client.delete_collection(self.collection)

        print(f"Creating collection '{self.collection}' (dim={EMBEDDING_DIM}, cosine)...")
        self.client.create_collection(
            collection_name=self.collection,
            vectors_config=VectorParams(size=EMBEDDING_DIM, distance=Distance.COSINE),
        )

    def collection_exists_and_populated(self) -> bool:
        """Check if collection exists and has vectors. Used to skip re-indexing."""
        try:
            info = self.client.get_collection(self.collection)
            return info.points_count > 0
        except Exception:
            return False

    def index_corpus(self, corpus: list[dict]) -> None:
        """
        Embed and upsert the corpus into Qdrant.

        We embed the CODE (not the docstring) because:
            - Queries are natural language
            - We want the model to bridge NL → code semantics
            - Embedding docstrings would make this too similar to BM25

        Args:
            corpus: list of dicts with 'id', 'code', 'docstring', 'url' keys.
        """
        if not self.collection_exists_and_populated():
            self._create_collection()
        else:
            print("Collection already populated — skipping indexing. Pass recreate_collection=True to re-index.")
            return

        print(f"Embedding {len(corpus)} documents (field: code_tokens)...")
        texts = [doc["code_tokens"] for doc in corpus]

        # SentenceTransformer.encode() handles batching internally,
        # but tqdm progress requires manual batching.
        all_vectors = []
        for i in tqdm(range(0, len(texts), _BATCH_SIZE), desc="Embedding batches"):
            batch = texts[i : i + _BATCH_SIZE]
            vecs = self.model.encode(batch, show_progress_bar=False, normalize_embeddings=True)
            all_vectors.extend(vecs.tolist())

        print("Upserting to Qdrant...")
        # Payload kept minimal (doc_id only). code/docstring/url are looked up
        # locally from the in-memory corpus at query time — Qdrant stores only
        # what's needed to identify the doc. Matches scripts/index_corpus.py.
        points = [
            PointStruct(
                id=i,
                vector=all_vectors[i],
                payload={"doc_id": corpus[i]["id"]},
            )
            for i in range(len(corpus))
        ]

        # Upsert in batches to avoid request size limits
        batch_size = 100
        for i in tqdm(range(0, len(points), batch_size), desc="Upserting"):
            self.client.upsert(
                collection_name=self.collection,
                points=points[i : i + batch_size],
            )

        print(f"Indexed {len(points)} vectors into '{self.collection}'.")

    def retrieve(self, query: str, top_k: int = 10, ef_search: int = 128) -> list[dict]:
        """
        Embed the query and search Qdrant for nearest neighbors.

        Args:
            query:     natural language query string
            top_k:     number of results to return
            ef_search: HNSW beam width. Higher = better recall, slower.
                       Experiment with this in M2 to build intuition.

        Returns:
            List of dicts: [{id, code, docstring, url, score}, ...], sorted by score desc.
        """
        query_vector = self.model.encode(
            query, normalize_embeddings=True
        ).tolist()

        results = self.client.query_points(
            collection_name=self.collection,
            query=query_vector,
            limit=top_k,
            search_params={"hnsw_ef": ef_search},
        )

        # Payload is doc_id only — see index_corpus(). Downstream callers that
        # need code/docstring/url look them up in the in-memory corpus by id.
        return [
            {"id": r.payload["doc_id"], "score": r.score}
            for r in results.points
        ]

    def _embed_queries_with_cache(self, queries: list[str]) -> list[list[float]]:
        """Return embedding vectors for queries, reusing the on-disk cache when present.

        All-hit (typical eval path) skips model.encode entirely. Partial-hit encodes
        only the missing texts and splices them into the right positions of the
        returned list. No-cache falls back to a single batch encode with a tqdm bar.
        """
        if not self._query_cache:
            return self.model.encode(
                queries,
                normalize_embeddings=True,
                show_progress_bar=True,
                batch_size=_BATCH_SIZE,
            ).tolist()

        miss_idx = [i for i, q in enumerate(queries) if q not in self._query_cache]
        n_hit = len(queries) - len(miss_idx)

        if not miss_idx:
            print(f"[dense] {n_hit:,} queries served from cache (skipped encode)")
            return [self._query_cache[q].tolist() for q in queries]

        print(
            f"[dense] query cache: {n_hit:,} hit / {len(miss_idx):,} miss — "
            f"encoding misses"
        )
        miss_texts = [queries[i] for i in miss_idx]
        miss_vecs = self.model.encode(
            miss_texts,
            normalize_embeddings=True,
            show_progress_bar=True,
            batch_size=_BATCH_SIZE,
        )

        out: list[list[float]] = [None] * len(queries)  # type: ignore[list-item]
        for i, q in enumerate(queries):
            if q in self._query_cache:
                out[i] = self._query_cache[q].tolist()
        for pos, vec in zip(miss_idx, miss_vecs):
            out[pos] = vec.tolist()
        return out

    def _query_batch_with_retry(self, requests: list, retries: int = 5):
        """Qdrant batch search with exponential backoff.

        Qdrant Cloud is reached over the network, and the local systemd-resolved
        stub can momentarily fail a DNS lookup under heavy CPU load ("Temporary
        failure in name resolution"). The upsert path already retries; without the
        same guard here, one transient blip aborts a whole 22k-query eval.
        """
        for attempt in range(retries):
            try:
                return self.client.query_batch_points(
                    collection_name=self.collection,
                    requests=requests,
                )
            except Exception as e:
                if attempt == retries - 1:
                    raise
                wait = min(2 ** attempt, 10) + random.random()
                tqdm.write(
                    f"[dense] Qdrant query retry {attempt + 1}/{retries} "
                    f"in {wait:.1f}s after: {e}"
                )
                time.sleep(wait)

    def retrieve_batch(
        self, queries: list[str], top_k: int = 100, ef_search: int = 128
    ) -> list[list[dict]]:
        """
        Embed all queries at once and search Qdrant for each.

        Batch-encoding is faster than calling retrieve() in a loop because
        sentence-transformers can parallelize the forward pass across the batch.
        The eval harness uses this fast path when available.

        Args:
            queries:   list of natural language query strings
            top_k:     results per query (default 100 for Recall@100 eval)
            ef_search: HNSW beam width (higher = better recall, slower)

        Returns:
            List of result lists, one per query. Each result list is
            [{id, code, docstring, url, score}, ...] sorted by score desc.
        """
        query_vectors = self._embed_queries_with_cache(queries)

        _SEARCH_BATCH = 50  # queries per HTTP request to Qdrant

        all_results = []
        total_batches = (len(query_vectors) + _SEARCH_BATCH - 1) // _SEARCH_BATCH
        for i in tqdm(range(0, len(query_vectors), _SEARCH_BATCH), total=total_batches, desc="Searching Qdrant"):
            chunk = query_vectors[i : i + _SEARCH_BATCH]
            requests = [
                QueryRequest(
                    query=qvec,
                    limit=top_k,
                    params=SearchParams(hnsw_ef=ef_search),
                    with_payload=True,
                )
                for qvec in chunk
            ]
            responses = self._query_batch_with_retry(requests)
            for resp in responses:
                all_results.append(
                    [
                        {"id": r.payload["doc_id"], "score": r.score}
                        for r in resp.points
                    ]
                )
        return all_results
