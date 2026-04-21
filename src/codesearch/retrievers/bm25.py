"""
BM25 retriever wrapping rank_bm25.

Lucene analogy:
    - BM25Retriever.__init__()  ≈  IndexWriter: tokenizes and builds the inverted index
    - BM25Retriever.retrieve()  ≈  IndexSearcher.search(): scores and returns TopDocs
    - tokenize()                ≈  Analyzer.tokenStream(): lowercases + whitespace splits

We're using docstrings as the indexed field (not code tokens) because:
    - CodeSearchNet queries are natural-language descriptions
    - BM25 on docstrings is the fair baseline; BM25 on raw code would be noisy
    - This is the same choice the original CodeSearchNet paper makes

In M3 you'll see that dense retrieval indexes the code itself via embeddings —
that asymmetry is part of why dense wins on this dataset.
"""

from __future__ import annotations

import re

from rank_bm25 import BM25Okapi
from tqdm import tqdm


def _tokenize(text: str) -> list[str]:
    """
    Minimal tokenizer: lowercase + split on non-alphanumeric.
    Equivalent to Lucene's StandardAnalyzer without stopword removal.
    Stopwords hurt BM25 on code queries, so we skip them.
    """
    return re.split(r"[^a-z0-9]+", text.lower())


class BM25Retriever:
    def __init__(self, corpus: list[dict]) -> None:
        """
        Index the corpus. Indexes the docstring field.

        Args:
            corpus: list of dicts with at least 'id' and 'docstring' keys.
        """
        self.corpus = corpus
        self._id_to_idx: dict[str, int] = {doc["id"]: i for i, doc in enumerate(corpus)}

        print("Tokenizing corpus for BM25...")
        tokenized = [_tokenize(doc["docstring"]) for doc in tqdm(corpus)]
        self.index = BM25Okapi(tokenized)
        print(f"BM25 index built over {len(corpus)} docs.")

    def retrieve(self, query: str, top_k: int = 10) -> list[dict]:
        """
        Retrieve top-k documents for a query.

        Returns:
            List of dicts: [{id, code, docstring, url, score}, ...], sorted by score desc.
        """
        tokens = _tokenize(query)
        scores = self.index.get_scores(tokens)

        # argsort descending — equivalent to Lucene's PriorityQueue<ScoreDoc>
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]

        return [
            {**self.corpus[i], "score": float(scores[i])}
            for i in top_indices
        ]
