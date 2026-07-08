"""
Offline corpus indexing — embed + upsert in parallel with a local vector cache.

Pipeline:
                                          ┌─► upsert worker 1 ┐
    embed thread ──► bounded queue ───────┼─► upsert worker 2 ├─► Qdrant
        (writes .npy memmap in-line)      └─► upsert worker 3 ┘

The embed thread is CPU-bound (sentence-transformers releases the GIL during
torch ops). The upsert workers are I/O-bound on the Qdrant HTTP API. They run
truly in parallel; total wall time ≈ max(embed_time, upsert_time / N_WORKERS).

Local cache:
    .cache/corpus_vectors_<model>.npy        memmap, shape (n_docs, dim), float32
    .cache/corpus_vectors_<model>.progress   single int — docs durably embedded

On re-run:
    - Embed picks up at sidecar progress (no cache → 0).
    - Upsert resumes from Qdrant's points_count.
    - When the cache is full but Qdrant is empty (cluster-reap recovery), no
      embedding happens — vectors stream from disk straight to upsert workers.
      ~3-5 min instead of ~15-30.
    - Upserts are idempotent on id, so re-pushing a cached batch that Qdrant
      already has is a no-op (re-overwrites the same point).

Progress bars:
    Two stacked tqdm bars — embed (top) and upsert (bottom) — update independently.

Usage:
    uv run python scripts/index_corpus.py            # build/resume the index
    uv run python scripts/index_corpus.py --recreate # drop and re-upsert (keeps cache)
"""

from __future__ import annotations

import argparse
import os
import queue
import random
import sys
import threading
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

import numpy as np
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

_BATCH_SIZE = 256
_QUEUE_MAX = 20          # backpressure: blocks producer if upsert lags this far behind
_UPSERT_WORKERS = 3
_CACHE_DIR = ".cache"
_UPSERT_RETRIES = 5


def _cache_paths(model: str) -> tuple[str, str]:
    safe = model.replace("/", "_")
    return (
        os.path.join(_CACHE_DIR, f"corpus_vectors_{safe}.npy"),
        os.path.join(_CACHE_DIR, f"corpus_vectors_{safe}.progress"),
    )


def _read_progress(path: str) -> int:
    if not os.path.exists(path):
        return 0
    try:
        with open(path) as f:
            return int((f.read() or "0").strip())
    except (ValueError, OSError):
        return 0


def _write_progress(path: str, n: int) -> None:
    """Atomic write so a crash mid-write never leaves a half-written number."""
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        f.write(str(n))
    os.replace(tmp, path)


def _open_memmap(path: str, n_rows: int, dim: int) -> np.memmap:
    """Open the vector cache as a memmap; create + pre-allocate if absent."""
    if not os.path.exists(path):
        return np.memmap(path, dtype=np.float32, mode="w+", shape=(n_rows, dim))
    expected = n_rows * dim * 4
    actual = os.path.getsize(path)
    if actual != expected:
        raise RuntimeError(
            f"Cache size mismatch at {path}: {actual} bytes on disk, expected "
            f"{expected} ({n_rows:,} × {dim} × 4). Delete it (and the .progress "
            f"sidecar) to recreate."
        )
    return np.memmap(path, dtype=np.float32, mode="r+", shape=(n_rows, dim))


def _upsert_worker(
    work_q: "queue.Queue",
    client: QdrantClient,
    bar: tqdm,
    bar_lock: threading.Lock,
) -> None:
    while True:
        item = work_q.get()
        if item is None:
            return
        offset, doc_ids, vectors = item
        points = [
            PointStruct(
                id=offset + j,
                vector=vectors[j].tolist(),
                payload={"doc_id": doc_ids[j]},
            )
            for j in range(len(doc_ids))
        ]
        for attempt in range(_UPSERT_RETRIES):
            try:
                client.upsert(collection_name=QDRANT_COLLECTION, points=points)
                break
            except Exception as e:
                if attempt == _UPSERT_RETRIES - 1:
                    raise
                wait = 2 ** attempt + random.random()
                with bar_lock:
                    tqdm.write(f"[upsert] retry in {wait:.1f}s after: {e}")
                time.sleep(wait)
        with bar_lock:
            bar.update(len(points))


def _produce(
    corpus: list[dict],
    model: SentenceTransformer | None,
    vectors_mmap: np.memmap,
    progress_path: str,
    embed_progress: int,
    upsert_start: int,
    work_q: "queue.Queue",
    embed_bar: tqdm,
    bar_lock: threading.Lock,
) -> None:
    n = len(corpus)

    # Phase 1: any docs already in the cache but not yet in Qdrant.
    # Stream them from disk straight to the upsert queue, no embedding.
    if upsert_start < embed_progress:
        for i in range(upsert_start, embed_progress, _BATCH_SIZE):
            end = min(i + _BATCH_SIZE, embed_progress)
            doc_ids = [corpus[k]["id"] for k in range(i, end)]
            # Copy out of the memmap so the worker isn't holding a slice that
            # could be invalidated when the file is closed.
            vecs = np.array(vectors_mmap[i:end], copy=True)
            work_q.put((i, doc_ids, vecs))

    # Phase 2: embed remaining docs, write to cache, push to queue.
    if embed_progress < n:
        assert model is not None, "Model required when embed_progress < n"
        for i in range(embed_progress, n, _BATCH_SIZE):
            end = min(i + _BATCH_SIZE, n)
            batch = corpus[i:end]
            texts = [doc["code_tokens"] for doc in batch]
            vecs = model.encode(
                texts, show_progress_bar=False, normalize_embeddings=True
            ).astype(np.float32)

            # Durably persist: write into memmap, flush, then bump sidecar.
            vectors_mmap[i:end] = vecs
            vectors_mmap.flush()
            _write_progress(progress_path, end)

            with bar_lock:
                embed_bar.update(end - i)

            work_q.put((i, [d["id"] for d in batch], vecs))

    # Sentinel one per worker — clean shutdown.
    for _ in range(_UPSERT_WORKERS):
        work_q.put(None)


def build_index(recreate: bool) -> None:
    # -------------------------------------------------------------------------
    # 1. Corpus
    # -------------------------------------------------------------------------
    corpus, _ = load_codesearch(n=-1)
    n = len(corpus)
    print(f"Corpus: {n:,} docs")

    # -------------------------------------------------------------------------
    # 2. Vector cache
    # -------------------------------------------------------------------------
    os.makedirs(_CACHE_DIR, exist_ok=True)
    vec_path, prog_path = _cache_paths(EMBEDDING_MODEL)
    vectors_mmap = _open_memmap(vec_path, n, EMBEDDING_DIM)
    embed_progress = min(_read_progress(prog_path), n)
    print(f"Cache : {embed_progress:,}/{n:,} vectors already embedded  ({vec_path})")

    # -------------------------------------------------------------------------
    # 3. Qdrant collection + resume offset
    # -------------------------------------------------------------------------
    print(f"Connecting to Qdrant at {QDRANT_URL} ...")
    client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY, timeout=60)

    existing = [c.name for c in client.get_collections().collections]
    if recreate and QDRANT_COLLECTION in existing:
        print(f"Dropping collection '{QDRANT_COLLECTION}' (--recreate).")
        client.delete_collection(QDRANT_COLLECTION)
        existing = [c for c in existing if c != QDRANT_COLLECTION]

    if QDRANT_COLLECTION not in existing:
        print(f"Creating collection '{QDRANT_COLLECTION}' (dim={EMBEDDING_DIM}, cosine).")
        client.create_collection(
            collection_name=QDRANT_COLLECTION,
            vectors_config=VectorParams(size=EMBEDDING_DIM, distance=Distance.COSINE),
        )
        upsert_start = 0
    else:
        upsert_start = client.get_collection(QDRANT_COLLECTION).points_count or 0

    print(f"Qdrant: {upsert_start:,}/{n:,} points already upserted")

    if embed_progress >= n and upsert_start >= n:
        print("Both cache and Qdrant are full — nothing to do.")
        return

    # -------------------------------------------------------------------------
    # 4. Load model only if we still need to embed
    # -------------------------------------------------------------------------
    if embed_progress < n:
        print(f"Loading model: {EMBEDDING_MODEL}")
        model = SentenceTransformer(EMBEDDING_MODEL)
    else:
        model = None
        print("Embedding step skipped — cache is complete.")

    # -------------------------------------------------------------------------
    # 5. Run the parallel pipeline
    # -------------------------------------------------------------------------
    bar_lock = threading.Lock()
    work_q: queue.Queue = queue.Queue(maxsize=_QUEUE_MAX)

    embed_bar = tqdm(total=n, initial=embed_progress, desc="Embed ", position=0, unit="doc")
    upsert_bar = tqdm(total=n, initial=upsert_start, desc="Upsert", position=1, unit="doc")

    t0 = time.perf_counter()

    workers = [
        threading.Thread(
            target=_upsert_worker,
            args=(work_q, client, upsert_bar, bar_lock),
            daemon=True,
            name=f"upsert-{i}",
        )
        for i in range(_UPSERT_WORKERS)
    ]
    for w in workers:
        w.start()

    try:
        _produce(
            corpus, model, vectors_mmap, prog_path,
            embed_progress, upsert_start,
            work_q, embed_bar, bar_lock,
        )
        for w in workers:
            w.join()
    finally:
        embed_bar.close()
        upsert_bar.close()
        # Move cursor below both bars before printing summary.
        print()

    elapsed = time.perf_counter() - t0
    final = client.get_collection(QDRANT_COLLECTION).points_count or 0
    print(f"Done in {elapsed:.0f}s.  Collection '{QDRANT_COLLECTION}' now has {final:,} vectors.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Embed the corpus and upsert into Qdrant (parallel + cached)."
    )
    parser.add_argument(
        "--recreate",
        action="store_true",
        help="Drop the Qdrant collection before indexing. Keeps the local "
             "vector cache — re-upsert is fast.",
    )
    args = parser.parse_args()
    build_index(recreate=args.recreate)


if __name__ == "__main__":
    main()
