"""
CodeSearchNet Python split loader.

Schema after loading:
    {
        "corpus":  List[dict]  — each dict has keys: id, code, docstring, url
        "queries": List[dict]  — each dict has keys: id, query, relevant_id
    }

The "document" is the code function.
The "query" is a natural-language description (docstring-derived).
Each query has exactly one relevant document (relevant_id maps to corpus id).

In smoke-test mode (n > 0), we load only n corpus docs and filter queries
to those whose relevant doc is in that subset — so eval still makes sense.
"""

from __future__ import annotations

from datasets import load_dataset
from tqdm import tqdm

from codesearch.config import SMOKE_TEST_SIZE


def load_codesearch(n: int = SMOKE_TEST_SIZE) -> tuple[list[dict], list[dict]]:
    """
    Load CodeSearchNet Python split.

    Args:
        n: Number of corpus docs to load. -1 = full corpus (~400k).
           Defaults to SMOKE_TEST_SIZE from config (100 for M0).

    Returns:
        (corpus, queries) — see module docstring for schema.
    """
    print(f"Loading CodeSearchNet Python split (n={'all' if n == -1 else n})...")

    # The 'train' split is the corpus. 'test' split has the eval queries.
    raw_corpus = load_dataset(
        "code_search_net", "python", split="train", trust_remote_code=True
    )
    raw_queries = load_dataset(
        "code_search_net", "python", split="test", trust_remote_code=True
    )

    # Build corpus — id is the repo+filepath+line, which is stable and unique.
    corpus: list[dict] = []
    seen_ids: set[str] = set()

    iterable = raw_corpus if n == -1 else raw_corpus.select(range(min(n, len(raw_corpus))))
    for row in tqdm(iterable, desc="Building corpus"):
        doc_id = row["func_name"]  # unique enough for our purposes
        if doc_id in seen_ids:
            continue
        seen_ids.add(doc_id)
        corpus.append(
            {
                "id": doc_id,
                "code": row["whole_func_string"],
                "docstring": row["func_documentation_string"],
                "url": row.get("func_code_url", ""),
            }
        )

    corpus_ids = {doc["id"] for doc in corpus}

    # Build queries — filter to those whose relevant doc is in our corpus subset.
    queries: list[dict] = []
    for row in tqdm(raw_queries, desc="Building queries"):
        relevant_id = row["func_name"]
        if relevant_id not in corpus_ids:
            continue
        queries.append(
            {
                "id": f"q_{row['func_name']}",
                "query": row["func_documentation_string"],
                "relevant_id": relevant_id,
            }
        )

    print(f"Loaded {len(corpus)} corpus docs, {len(queries)} eval queries.")
    return corpus, queries
