#!/usr/bin/env python3
"""Wolf Defender (patronus-studio), multilingual ModernBERT.

    python3 -m quadrat.run --detector wolfdefender

Second on the indirect-injection leaderboard of `pi-detector-bench` (AUC 0.865), behind Bastion
and ahead of every DeBERTa classifier in this table.

ITS APERTURE IS 8192 TOKENS, not 512, and that is the point worth getting right. It is a ModernBERT
(mmBERT) model with `max_position_embeddings = 8192`, so roughly 32000 characters -- it reads a
whole document of this corpus in one pass, where ProtectAI and PromptGuard see about 2000
characters at a time. Handing it the 512-token window would measure our splitter instead of the
model, and would report a long-context detector as if it were a short-context one. The declared
limit here is its own; the harness splits behind it only for documents that genuinely exceed it.

POLARITY IS FROM THE MODEL CARD, not from a guess: the checkpoint ships no `id2label`, and the card
states "classifies inputs into benign (0) and injection-detected (1)". Getting this backwards
produces a complete, plausible, exactly inverted table, which is why the index is written down with
its source rather than left to `fallback_index`.

Trained on "a mixture of publicly available prompt injection datasets and internally generated
examples", and this checkpoint on ~5% of that data (~50000 rows) by the authors' own note. Public
material means the phrasing of published attack families may be familiar to it; our positives were
generated fresh, so they cannot be in it.
"""
from __future__ import annotations

import pathlib

from ..detector import register
from ._hf import _HFClassifier


@register("wolfdefender", version="modernbert-0.3B")
class WolfDefender(_HFClassifier):
    display = "Wolf Defender (ModernBERT 0.3B)"
    model_id = "patronus-studio/wolf-defender-prompt-injection"
    #: pinned: hub state read 2026-08-09
    revision = "ecc382bd4d98ffa19e1c9c2ce4a0722904c04a3c"
    #: 8192 tokens at the 4 characters per token used here. Its own limit, not ours.
    max_chars = 8192 * 4
    #: no id2label in the checkpoint; index 1 is "injection-detected" per the model card
    fallback_index = 1
    #: the forward must see the whole window, or the aperture above is a claim about nothing
    max_tokens = 8192
    #: 8192-token sequences: a batch of 32 does not fit beside the two passes already on the card
    batch = 4

    def _load(self):
        """The tokenizer needs building by hand, and the reason is worth recording.

        `tokenizer_config.json` names `tokenizer_class: TokenizersBackend` -- a class this
        transformers does not have, because the repo was saved by a newer one. `AutoTokenizer`
        then refuses the whole repo. The artefact itself is an ordinary `tokenizer.json`, so it is
        loaded as the fast tokenizer it is, with the special tokens taken from the same config.
        Nothing about the model is changed by this; only the loader is."""
        from transformers import (AutoModelForSequenceClassification, AutoConfig,
                                  PreTrainedTokenizerFast)
        from huggingface_hub import hf_hub_download
        import json

        model = AutoModelForSequenceClassification.from_pretrained(
            self.model_id, revision=self.revision, trust_remote_code=False)
        tok_file = hf_hub_download(self.model_id, "tokenizer.json", revision=self.revision)
        cfg_file = hf_hub_download(self.model_id, "tokenizer_config.json",
                                   revision=self.revision)
        cfg = json.loads(pathlib.Path(cfg_file).read_text(encoding="utf-8"))
        keep = {k: v for k, v in cfg.items()
                if k.endswith("_token") and isinstance(v, str)}
        tok = PreTrainedTokenizerFast(tokenizer_file=tok_file,
                                      model_max_length=cfg.get("model_max_length", 8192),
                                      **keep)
        return model, tok
