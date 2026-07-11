"""Encoder factory + a faithful UniXcoder wrapper (M5 sub-experiment B).

The pipeline (scripts/index_corpus.py, retrievers/dense.py) embeds text through a
single object exposing SentenceTransformer's `.encode(texts, normalize_embeddings=...)`
surface. Most models load fine as a plain SentenceTransformer. UniXcoder does NOT:
its official code-search recipe frames the input with a mode token that the BPE
tokenizer only recognises when inserted as a raw id, so ST's default path uses the
model off-distribution (a degenerate encoding). `UniXcoderEncoder` reproduces the
official recipe so the "fair code model" comparison actually tests UniXcoder, not
our misuse of it.

Official recipe (microsoft/CodeBERT · UniXcoder, encoder-only mode):
    ids  = [<s>, <encoder-only>, </s>] + bpe(text)[:max_length-4] + [</s>]
    pool = mean over non-pad tokens (specials included, exactly as upstream)
    emb  = L2-normalize(pool)
Code docs and NL queries go through this identical path (symmetric bi-encoder).
"""

from __future__ import annotations

import numpy as np
import torch


class UniXcoderEncoder:
    """Faithful reproduction of UniXcoder's encoder-only code-search encoding.

    Exposes the subset of `SentenceTransformer.encode()` the pipeline uses:
    accepts a str or list[str] plus `normalize_embeddings`, `batch_size`,
    `show_progress_bar` (the last ignored), and returns an np.ndarray
    (1-D for a single string, else 2-D). Extra kwargs are ignored for
    drop-in compatibility.
    """

    def __init__(
        self,
        model_name: str = "microsoft/unixcoder-base",
        max_length: int = 512,
        device: str | None = None,
    ) -> None:
        from transformers import AutoModel, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.eval()
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.max_length = max_length

        self.cls_id = self.tokenizer.cls_token_id  # <s> = 0
        self.sep_id = self.tokenizer.sep_token_id  # </s> = 2
        self.pad_id = self.tokenizer.pad_token_id  # <pad> = 1
        self.mode_id = self.tokenizer.convert_tokens_to_ids("<encoder-only>")  # 6
        assert self.mode_id != self.tokenizer.unk_token_id, (
            "<encoder-only> is not a known special token for this tokenizer"
        )

    def _build_ids(self, text: str) -> list[int]:
        toks = self.tokenizer.tokenize(text)[: self.max_length - 4]
        ids = self.tokenizer.convert_tokens_to_ids(toks)
        return [self.cls_id, self.mode_id, self.sep_id] + ids + [self.sep_id]

    @torch.no_grad()
    def encode(
        self,
        texts,
        batch_size: int = 64,
        normalize_embeddings: bool = True,
        show_progress_bar: bool = False,  # accepted, ignored (drop-in compat)
        convert_to_numpy: bool = True,
        **kwargs,
    ) -> np.ndarray:
        single = isinstance(texts, str)
        if single:
            texts = [texts]

        out: list[np.ndarray] = []
        for i in range(0, len(texts), batch_size):
            seqs = [self._build_ids(t) for t in texts[i : i + batch_size]]
            maxlen = max(len(s) for s in seqs)
            input_ids = torch.full(
                (len(seqs), maxlen), self.pad_id, dtype=torch.long
            )
            for j, s in enumerate(seqs):
                input_ids[j, : len(s)] = torch.tensor(s, dtype=torch.long)
            input_ids = input_ids.to(self.device)
            mask = input_ids.ne(self.pad_id)

            hidden = self.model(input_ids, attention_mask=mask)[0]
            m = mask[:, :, None].float()
            emb = (hidden * m).sum(1) / m.sum(1).clamp(min=1e-9)
            if normalize_embeddings:
                emb = torch.nn.functional.normalize(emb, p=2, dim=1)
            out.append(emb.cpu().numpy().astype(np.float32))

        arr = np.concatenate(out, axis=0)
        return arr[0] if single else arr


def load_encoder(model_name: str, **kwargs):
    """Return an encoder exposing `.encode(texts, normalize_embeddings=...)`.

    UniXcoder needs its official mode-token framing (UniXcoderEncoder); everything
    else loads as a SentenceTransformer with trust_remote_code so custom-arch code
    models still work.
    """
    if "unixcoder" in model_name.lower():
        from codesearch.config import EMBED_MAX_LENGTH

        kwargs.setdefault("max_length", EMBED_MAX_LENGTH)
        return UniXcoderEncoder(model_name, **kwargs)
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(model_name, trust_remote_code=True)
