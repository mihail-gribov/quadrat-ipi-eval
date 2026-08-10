#!/usr/bin/env python3
"""ProtectAI's prompt-injection classifier (deberta-v3-base, v2).

    python3 -m quadrat.run --detector protectai --window --policy chunk

512 tokens, so the window is not really optional: without it the run measures truncation
rather than detection, and 93% of the doc carrier is longer than that."""
from __future__ import annotations

from ..detector import register
from ._hf import _HFClassifier


@register("protectai", version="deberta-v3-base-v2")
class ProtectAI(_HFClassifier):
    display = "ProtectAI DeBERTa v3 base v2"
    model_id = "protectai/deberta-v3-base-prompt-injection-v2"
    positive_label = "INJECTION"
