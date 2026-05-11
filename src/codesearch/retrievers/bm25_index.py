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

import os
import pickle

import bm25s

_CORPUS_FILENAME = "corpus_dicts.pkl"


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

    @staticmethod
    def exists(cache_dir: str) -> bool:
        """Cheap check: does cache_dir look like a complete BM25Index save?"""
        return os.path.isdir(cache_dir) and os.path.exists(
            os.path.join(cache_dir, _CORPUS_FILENAME)
        )
