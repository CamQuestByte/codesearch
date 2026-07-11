"""
Central config. All env vars live here — nothing else imports dotenv directly.
"""

import os
from dotenv import load_dotenv

load_dotenv()

# Quiet the transformers "weight not used / UNEXPECTED" table that fires every
# time SentenceTransformer loads MiniLM. The unused key is `embeddings.position_ids`,
# a buffer the newer transformers release dropped from the BertModel state dict; the
# message is informational and safe to ignore for this checkpoint.
import transformers  # noqa: E402
transformers.logging.set_verbosity_error()


# --- Qdrant ---
QDRANT_URL: str = os.environ["QDRANT_URL"]
QDRANT_API_KEY: str = os.environ["QDRANT_API_KEY"]
QDRANT_COLLECTION: str = os.getenv("QDRANT_COLLECTION", "codesearch-minilm")

# --- Embedding ---
EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
# 384 for MiniLM-L6-v2; override via env for a different model (e.g. 768 for
# jina-embeddings-v2-base-code). Must match the Qdrant collection's vector size.
EMBEDDING_DIM: int = int(os.getenv("EMBEDDING_DIM", "384"))

# What text we embed per doc for dense retrieval (M5 experiment):
#   "code_tokens"   — CSN space-joined tokens (default; the M0-M4 setup)
#   "code_stripped" — docstring-stripped function source (data.strip_docstring)
# Set alongside a matching QDRANT_COLLECTION when indexing a variant.
EMBED_INPUT: str = os.getenv("EMBED_INPUT", "code_tokens")

# Max wordpiece length for encoders we drive manually (UniXcoderEncoder). Default
# 256 to match all-MiniLM-L6-v2's max_seq_length (the M0-M4 baseline), so an M5
# model swap is an apples-to-apples comparison at the same truncation — not a model
# that silently gets 2x the context. Also caps O(L^2) attention cost. SentenceTransformer
# models ignore this (their own max_seq_length governs). See [[todo-embed-raw-code]].
EMBED_MAX_LENGTH: int = int(os.getenv("EMBED_MAX_LENGTH", "256"))

# Store vectors on disk (Qdrant on_disk=True) instead of RAM. Needed when a new
# collection would push total Qdrant RAM past the 1GB free tier — e.g. a second
# 384-dim collection, or any 768-dim model over 434k docs.
QDRANT_ON_DISK: bool = os.getenv("QDRANT_ON_DISK", "false").lower() in ("1", "true", "yes")

# --- Dataset ---
# SMOKE_TEST_SIZE: how many docs to load for M0 hello-world.
# Set to -1 in .env (or in the HF Spaces secret) to load the full corpus.
_smoke = os.getenv("SMOKE_TEST_SIZE", "100")
SMOKE_TEST_SIZE: int = int(_smoke)

# --- Retrieval defaults ---
TOP_K: int = 10       # results returned to the UI
RECALL_K: int = 100   # candidate pool for eval (Recall@100)
