"""
BM25 index: a persistable wrapper around bm25s.BM25 + the corpus payload.

Lucene analogy:
    BM25Index           ≈ on-disk Index (Lucene Directory + IndexReader)
    BM25Index.build()   ≈ IndexWriter
    BM25Index.save()    ≈ commit() / sync to disk
    BM25Index.load()    ≈ DirectoryReader.open()
    BM25Retriever       ≈ IndexSearcher (uses an index to score queries)

The bm25s native files (.npy / .json) capture the inverted index and IDF
statistics. We pickle the corpus dicts alongside so that retrieve() can return
rich result payloads (id, code, docstring, url) without re-loading the HF
dataset on every script run.

Disk layout (one directory per index):
    {cache_dir}/
        params.json
        data.csc.index.npy
        ...                     # bm25s native files
        corpus_dicts.pkl        # our per-doc payload, indexed positionally
"""

from __future__ import annotations

import hashlib
import os
import pickle

import bm25s

_CORPUS_FILENAME = "corpus_dicts.pkl"
_ID_FINGERPRINT = "id_order.sha256"


def _hash_ids(corpus: list[dict]) -> str:
    """SHA-256 over the corpus doc-id sequence.

    bm25s scores by POSITIONAL index — retrieve() returns row numbers into the
    corpus. A slim index (shipped without its corpus payload) is therefore only
    valid against a corpus in the exact same order it was built from. This
    fingerprint turns that invariant into a checked one: load_index_only()
    refuses a corpus whose id-order doesn't match, so misalignment fails loud
    instead of silently mis-ranking.
    """
    h = hashlib.sha256()
    for doc in corpus:
        h.update(doc["id"].encode("utf-8"))
        h.update(b"\n")
    return h.hexdigest()


class BM25Index:
    """Holds an in-memory BM25 index + its corpus, with save/load."""

    def __init__(self, index: bm25s.BM25, corpus: list[dict]) -> None:
        self.index = index
        self.corpus = corpus

    @classmethod
    def build(cls, corpus: list[dict]) -> "BM25Index":
        """Tokenize the corpus and build a fresh BM25 index."""
        print("Tokenizing corpus for BM25...")
        code = [doc["code_tokens"] for doc in corpus]
        corpus_tokens = bm25s.tokenize(code, stopwords=None, show_progress=True)

        index = bm25s.BM25()
        index.index(corpus_tokens)
        print(f"BM25 index built over {len(corpus):,} docs.")
        return cls(index, corpus)

    def save(self, cache_dir: str) -> None:
        """Write the index + corpus payload to a directory."""
        os.makedirs(cache_dir, exist_ok=True)
        self.index.save(cache_dir)
        with open(os.path.join(cache_dir, _CORPUS_FILENAME), "wb") as f:
            pickle.dump(self.corpus, f, protocol=pickle.HIGHEST_PROTOCOL)

    def save_index_only(self, cache_dir: str) -> None:
        """Write ONLY the bm25s native files + an id-order fingerprint.

        Skips the large corpus pickle that save() writes. The corpus is
        re-supplied at load time from wherever it already lives in memory
        (on the Space, the HF-dataset corpus loaded at boot). This is the
        ~150MB artifact shipped to the HF Space via LFS, vs. save()'s ~1.1GB.
        """
        os.makedirs(cache_dir, exist_ok=True)
        self.index.save(cache_dir)
        with open(os.path.join(cache_dir, _ID_FINGERPRINT), "w") as f:
            f.write(_hash_ids(self.corpus))

    @classmethod
    def load(cls, cache_dir: str) -> "BM25Index":
        """Load an index + corpus payload previously written by save()."""
        index = bm25s.BM25.load(cache_dir)
        corpus_path = os.path.join(cache_dir, _CORPUS_FILENAME)
        if not os.path.exists(corpus_path):
            raise FileNotFoundError(
                f"{corpus_path} missing — cache_dir was not produced by BM25Index.save()."
            )
        with open(corpus_path, "rb") as f:
            corpus = pickle.load(f)
        return cls(index, corpus)

    @classmethod
    def load_index_only(cls, cache_dir: str, corpus: list[dict]) -> "BM25Index":
        """Load a slim index (native files, no corpus pickle) and attach an
        externally-supplied corpus, verifying it matches the build-time order.

        Raises ValueError if the corpus id-order doesn't match the fingerprint
        written by save_index_only() — catching a silent mis-ranking where the
        positional index no longer lines up with the corpus rows.
        """
        index = bm25s.BM25.load(cache_dir)
        fingerprint_path = os.path.join(cache_dir, _ID_FINGERPRINT)
        if not os.path.exists(fingerprint_path):
            raise FileNotFoundError(
                f"{fingerprint_path} missing — cache_dir was not produced by "
                f"save_index_only()."
            )
        with open(fingerprint_path) as f:
            expected = f.read().strip()
        actual = _hash_ids(corpus)
        if actual != expected:
            raise ValueError(
                "Supplied corpus does not match the built BM25 index "
                f"(id-order fingerprint {actual[:12]}… != {expected[:12]}…). "
                "Rebuild the slim index, or check the corpus source / "
                "SMOKE_TEST_SIZE matches the build."
            )
        return cls(index, corpus)

    @staticmethod
    def exists(cache_dir: str) -> bool:
        """Cheap check: does cache_dir look like a complete BM25Index save?"""
        return os.path.isdir(cache_dir) and os.path.exists(
            os.path.join(cache_dir, _CORPUS_FILENAME)
        )
