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

from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct,
    SearchRequest,
)
from tqdm import tqdm

from codesearch.config import (
    QDRANT_URL,
    QDRANT_API_KEY,
    QDRANT_COLLECTION,
    EMBEDDING_MODEL,
    EMBEDDING_DIM,
)

# Batch size for embedding. MiniLM is small; 256 is fast on CPU.
# Reduce to 64 if you hit memory issues on HF Spaces.
_BATCH_SIZE = 256


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
        self.model = SentenceTransformer(EMBEDDING_MODEL)

        print(f"Connecting to Qdrant at {QDRANT_URL}")
        self.client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
        self.collection = QDRANT_COLLECTION

        if recreate_collection:
            self._create_collection()

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

        print(f"Embedding {len(corpus)} documents (field: code)...")
        texts = [doc["code"] for doc in corpus]

        # SentenceTransformer.encode() handles batching internally,
        # but tqdm progress requires manual batching.
        all_vectors = []
        for i in tqdm(range(0, len(texts), _BATCH_SIZE), desc="Embedding batches"):
            batch = texts[i : i + _BATCH_SIZE]
            vecs = self.model.encode(batch, show_progress_bar=False, normalize_embeddings=True)
            all_vectors.extend(vecs.tolist())

        print("Upserting to Qdrant...")
        points = [
            PointStruct(
                id=i,
                vector=all_vectors[i],
                payload={
                    "doc_id": corpus[i]["id"],
                    "docstring": corpus[i]["docstring"],
                    "code": corpus[i]["code"],
                    "url": corpus[i]["url"],
                },
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

        return [
            {
                "id": r.payload["doc_id"],
                "code": r.payload["code"],
                "docstring": r.payload["docstring"],
                "url": r.payload["url"],
                "score": r.score,
            }
            for r in results.points
        ]
