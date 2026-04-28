"""
CodeSearchNet Python split loader.

Schema after loading:
    {
        "corpus":  List[dict]  — each dict has keys: id, code, docstring, url
        "queries": List[dict]  — each dict has keys: id, query, relevant_id
    }

Document ID is `func_code_url` — globally unique (encodes repo + commit + file + lines).
Using `func_name` as ID is wrong: the same name (e.g. `parse`, `get`) appears in hundreds
of unrelated repos, and train/test splits come from completely different codebases.

Smoke-test mode (0 < n):
    Corpus  = first n docs from the test split.
    Queries = the non-empty docstrings from those same n docs.
    Every query has a valid relevant doc in the corpus. BM25 will score near-perfectly
    here (query == indexed docstring) — this mode only validates the pipeline, not quality.

Full-corpus mode (n == -1):
    Corpus  = train split (~412k) + test split (~22k) = ~434k total.
    Queries = test split docstrings (~22k, filtered for non-empty).
    Each test function IS in the corpus (URL uniqueness confirmed: zero overlap
    between train and test URLs), so every query has valid ground truth.
    BM25 must find the relevant doc among ~434k candidates.
"""

from __future__ import annotations

from datasets import load_dataset
from tqdm import tqdm

from codesearch.config import SMOKE_TEST_SIZE


def _make_doc(row: dict) -> dict:
    url = row["func_code_url"]
    return {
        "id": url,
        "code": row["whole_func_string"],
        "code_tokens": " ".join(row["func_code_tokens"]),
        "docstring": row["func_documentation_string"],
        "url": url,
    }


def load_codesearch(n: int = SMOKE_TEST_SIZE) -> tuple[list[dict], list[dict]]:
    """
    Load CodeSearchNet Python split.

    Args:
        n: -1 = full corpus (train + test, ~434k docs).
           n > 0 = smoke-test: first n docs from test split only.

    Returns:
        (corpus, queries) — see module docstring for schema.
    """
    raw_test = load_dataset("code_search_net", "python", split="test")

    corpus: list[dict] = []
    seen_urls: set[str] = set()

    if n != -1:
        print(f"Loading CodeSearchNet Python split (smoke-test, n={n})...")
        for row in tqdm(raw_test, desc="Building corpus"):
            if len(corpus) >= n:
                break
            url = row["func_code_url"]
            if not url or url in seen_urls:
                continue
            seen_urls.add(url)
            corpus.append(_make_doc(row))
    else:
        print("Loading CodeSearchNet Python split (full corpus)...")
        raw_train = load_dataset("code_search_net", "python", split="train")
        for split_name, raw in [("train", raw_train), ("test", raw_test)]:
            for row in tqdm(raw, desc=f"Building corpus ({split_name})"):
                url = row["func_code_url"]
                if not url or url in seen_urls:
                    continue
                seen_urls.add(url)
                corpus.append(_make_doc(row))

    corpus_ids = {doc["id"] for doc in corpus}

    # Queries: test-split docstrings whose function is in the corpus.
    # relevant_id = func_code_url, which is the doc["id"] of the target function.
    queries: list[dict] = []
    for row in tqdm(raw_test, desc="Building queries"):
        url = row["func_code_url"]
        if url not in corpus_ids:
            continue
        query_text = row["func_documentation_string"]
        if not query_text or not query_text.strip():
            continue
        queries.append(
            {
                "id": f"q_{url}",
                "query": query_text,
                "relevant_id": url,
            }
        )

    print(f"Loaded {len(corpus):,} corpus docs, {len(queries):,} eval queries.")
    return corpus, queries
