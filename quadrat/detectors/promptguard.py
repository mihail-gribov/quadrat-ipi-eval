#!/usr/bin/env python3
"""Meta's Llama Prompt Guard 2 (86M).

    python3 -m quadrat.run --detector promptguard --window --policy chunk

The meta-llama repo is gated. Without access this falls back to a mirror carrying the same
fp32 weights and tokenizer, so a machine without a token measures the same model instead of
skipping it -- and the substitution is recorded in the run's notes, not passed over."""
from __future__ import annotations

from ..detector import register
from ._hf import _HFClassifier


@register("promptguard", version="2-86M")
class PromptGuard(_HFClassifier):
    display = "Meta Prompt Guard 2 (86M)"
    # The meta-llama repo is gated; the mirror ships the same fp32 weights and tokenizer, so a
    # machine without repo access measures the same model rather than skipping it.
    model_id = "meta-llama/Llama-Prompt-Guard-2-86M"
    mirror_id = "gravitee-io/Llama-Prompt-Guard-2-86M-onnx"
    positive_label = "LABEL_1"
    fallback_index = 1

    def setup(self):
        try:
            super().setup()
        except Exception as e:                  # gated repo, no token, or offline
            self.model_id = self.mirror_id
            super().setup()
            self.notes += f" (mirror; original unavailable: {type(e).__name__})"
