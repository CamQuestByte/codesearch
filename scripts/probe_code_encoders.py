"""M5 sub-experiment B: probe fair code bi-encoders that load under current transformers.

For each candidate: try to load via SentenceTransformer (trust_remote_code where
needed), encode a short code snippet + a NL query, report embedding dim and a
sanity cosine. Failures are caught and reported per-model so one bad model does
not abort the sweep.
"""
import sys
import traceback

SNIPPET = "def add(a, b):\n    return a + b"
QUERY = "add two numbers"

# (model_id, needs_trust_remote_code, note)
CANDIDATES = [
    ("microsoft/unixcoder-base", False, "RoBERTa arch; may need mean-pooling wrapper"),
    ("microsoft/codebert-base", False, "RoBERTa arch; may need mean-pooling wrapper"),
    ("jinaai/jina-embeddings-v2-base-code", True, "known-blocked baseline (find_pruneable_heads_and_indices)"),
    ("nomic-ai/CodeRankEmbed", True, "newer code bi-encoder, standard-ish arch"),
    ("Alibaba-NLP/gte-modernbert-base", True, "modern general encoder, control"),
]


def probe_sentence_transformer(model_id, trust):
    import numpy as np
    from sentence_transformers import SentenceTransformer

    st = SentenceTransformer(model_id, trust_remote_code=trust)
    emb = st.encode([SNIPPET, QUERY], normalize_embeddings=True)
    dim = emb.shape[1]
    cos = float(np.dot(emb[0], emb[1]))
    return dim, cos


def probe_mean_pool(model_id):
    """Fallback: raw HF model + mean pooling (for encoders with no ST config)."""
    import numpy as np
    import torch
    from transformers import AutoModel, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModel.from_pretrained(model_id)
    model.eval()

    def encode(text):
        batch = tok(text, return_tensors="pt", truncation=True, max_length=256)
        with torch.no_grad():
            out = model(**batch).last_hidden_state
        mask = batch["attention_mask"].unsqueeze(-1).float()
        vec = (out * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
        vec = torch.nn.functional.normalize(vec, dim=-1)
        return vec[0].numpy()

    v_code = encode(SNIPPET)
    v_query = encode(QUERY)
    return v_code.shape[0], float(np.dot(v_code, v_query))


def main():
    results = []
    for model_id, trust, note in CANDIDATES:
        print(f"\n{'='*70}\n{model_id}  ({note})\n{'='*70}", flush=True)
        row = {"model": model_id, "note": note}
        try:
            dim, cos = probe_sentence_transformer(model_id, trust)
            row.update(status="OK (ST)", dim=dim, cos=round(cos, 4))
            print(f"  -> OK via SentenceTransformer: dim={dim} cos={cos:.4f}", flush=True)
        except Exception as e:
            print(f"  ST load failed: {type(e).__name__}: {e}", flush=True)
            print("  trying raw mean-pooling fallback...", flush=True)
            try:
                dim, cos = probe_mean_pool(model_id)
                row.update(status="OK (mean-pool)", dim=dim, cos=round(cos, 4))
                print(f"  -> OK via mean-pool: dim={dim} cos={cos:.4f}", flush=True)
            except Exception as e2:
                row.update(status=f"FAIL: {type(e2).__name__}", dim=None, cos=None)
                print(f"  -> FAIL: {type(e2).__name__}: {e2}", flush=True)
                traceback.print_exc()
        results.append(row)

    print(f"\n\n{'#'*70}\nSUMMARY\n{'#'*70}")
    for r in results:
        print(f"  {r['status']:>18}  dim={r['dim']}  cos={r['cos']}  {r['model']}")


if __name__ == "__main__":
    sys.exit(main())
