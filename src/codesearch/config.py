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
EMBEDDING_DIM: int = 384  # MiniLM-L6-v2 output dim. Update if you swap models in M5.

# --- Dataset ---
# SMOKE_TEST_SIZE: how many docs to load for M0 hello-world.
# Set to -1 in .env (or in the HF Spaces secret) to load the full corpus.
_smoke = os.getenv("SMOKE_TEST_SIZE", "100")
SMOKE_TEST_SIZE: int = int(_smoke)

# --- Retrieval defaults ---
TOP_K: int = 10       # results returned to the UI
RECALL_K: int = 100   # candidate pool for eval (Recall@100)
