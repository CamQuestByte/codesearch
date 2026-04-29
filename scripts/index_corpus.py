"""
Offline corpus indexing script — run once before using dense retrieval.

Embeds all corpus documents into Qdrant using the configured sentence-transformer
model. Uses func_code_tokens (docstrings stripped) as the indexed text, consistent
with the BM25 baseline, so differences in eval scores reflect retrieval method
not content differences.

Usage:
    # First run — build the index:
    uv run python scripts/index_corpus.py

    # Resume after an interruption (auto-detected, no flag needed):
    uv run python scripts/index_corpus.py

    # Re-index after a model change or corpus update:
    uv run python scripts/index_corpus.py --recreate

Resume logic:
    Embedding and upserting are interleaved — every batch is committed to Qdrant
    before the next one starts. On restart, the script checks points_count and
    skips the already-indexed prefix of the corpus.

Why run this offline and not at app boot?
    Embedding 434k docs takes ~10 min on CPU. Storing vectors in Qdrant means
    every subsequent query only needs to embed one string (~5ms). Running this
    at boot would make the app unusable for the first 10 minutes.
"""

from __future__ import annotations

import argparse
import sys
import os
import time
import random

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

from codesearch.config import (
    QDRANT_URL,
    QDRANT_API_KEY,
    QDRANT_COLLECTION,
    EMBEDDING_MODEL,
    EMBEDDING_DIM,
)
from codesearch.data import load_codesearch

_BATCH_SIZE = 128  # docs per embed+upsert step — smaller = less likely to timeout on Qdrant Cloud


def build_index(recreate: bool) -> None:
    # -------------------------------------------------------------------------
    # 1. Load corpus
    # -------------------------------------------------------------------------
    corpus, _ = load_codesearch(n=-1)
    print(f"Corpus: {len(corpus):,} docs")

    # -------------------------------------------------------------------------
    # 2. Connect to Qdrant, handle recreate / resume
    # -------------------------------------------------------------------------
    print(f"Connecting to Qdrant at {QDRANT_URL} ...")
    client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY, timeout=60)

    existing = [c.name for c in client.get_collections().collections]

    if recreate and QDRANT_COLLECTION in existing:
        print(f"Dropping existing collection '{QDRANT_COLLECTION}' ...")
        client.delete_collection(QDRANT_COLLECTION)
        existing = [c for c in existing if c != QDRANT_COLLECTION]

    if QDRANT_COLLECTION not in existing:
        print(f"Creating collection '{QDRANT_COLLECTION}' (dim={EMBEDDING_DIM}, cosine) ...")
        client.create_collection(
            collection_name=QDRANT_COLLECTION,
            vectors_config=VectorParams(size=EMBEDDING_DIM, distance=Distance.COSINE),
        )
        resume_from = 0
    else:
        resume_from = client.get_collection(QDRANT_COLLECTION).points_count or 0
        if resume_from >= len(corpus):
            print(f"Collection already fully indexed ({resume_from:,} points). Pass --recreate to re-index.")
            return
        if resume_from > 0:
            print(f"Resuming from doc {resume_from:,} ({resume_from:,} already indexed).")
        else:
            print("Collection exists but is empty — starting from scratch.")

    # -------------------------------------------------------------------------
    # 3. Load model
    # -------------------------------------------------------------------------
    print(f"Loading model: {EMBEDDING_MODEL}")
    model = SentenceTransformer(EMBEDDING_MODEL)

    # -------------------------------------------------------------------------
    # 4. Embed + upsert in interleaved batches
    #
    # Each batch is committed to Qdrant before the next one starts, so killing
    # the process at any point loses at most one batch (~256 docs).
    # On resume, we skip the already-indexed prefix via resume_from.
    # -------------------------------------------------------------------------
    remaining = corpus[resume_from:]
    print(f"Embedding and upserting {len(remaining):,} docs in batches of {_BATCH_SIZE} ...")

    t0 = time.perf_counter()
    total_done = resume_from

    for i in tqdm(range(0, len(remaining), _BATCH_SIZE), desc="Indexing"):
        batch = remaining[i : i + _BATCH_SIZE]

        vecs = model.encode(
            [doc["code_tokens"] for doc in batch],
            show_progress_bar=False,
            normalize_embeddings=True,
        )

        points = [
            PointStruct(
                id=total_done + j,
                vector=vecs[j].tolist(),
                payload={
                    "doc_id": batch[j]["id"],
                    "code": batch[j]["code"],
                    "docstring": batch[j]["docstring"],
                    "url": batch[j]["url"],
                },
            )
            for j in range(len(batch))
        ]

        for attempt in range(5):
            try:
                client.upsert(collection_name=QDRANT_COLLECTION, points=points)
                break
            except Exception as e:
                if attempt == 4:
                    raise
                wait = 2 ** attempt + random.random()
                print(f"\nUpsert failed ({e}), retrying in {wait:.1f}s ...")
                time.sleep(wait)
        total_done += len(batch)

    elapsed = time.perf_counter() - t0
    print(f"Done in {elapsed:.0f}s ({len(remaining)/elapsed:.0f} docs/s)")
    print(f"Collection '{QDRANT_COLLECTION}' ready with {total_done:,} vectors.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Embed corpus and load into Qdrant.")
    parser.add_argument(
        "--recreate",
        action="store_true",
        help="Drop and recreate the collection even if it already has vectors.",
    )
    args = parser.parse_args()
    build_index(recreate=args.recreate)


if __name__ == "__main__":
    main()
