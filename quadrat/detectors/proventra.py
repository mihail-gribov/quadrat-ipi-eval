#!/usr/bin/env python3
"""Proventra, mDeBERTa-v3-base prompt-injection classifier.

    python3 -m quadrat.run --detector proventra

Fifth on the indirect-injection leaderboard of `pi-detector-bench` (AUC 0.821), and the strongest
of the 512-token DeBERTa family there -- ahead of ProtectAI v2 (0.816), PromptGuard 2 86M (0.789)
and deepset (0.787). Multilingual by construction (mDeBERTa), which this corpus does not exercise:
every carrier here is English, so its multilingual head is measured on the one language it shares
with the others rather than on the axis it was built for.
"""
from __future__ import annotations

from ..detector import register
from ._hf import _HFClassifier


@register("proventra", version="mdeberta-v3-base")
class Proventra(_HFClassifier):
    display = "Proventra mDeBERTa v3 base"
    model_id = "proventra/mdeberta-v3-base-prompt-injection"
    #: pinned: hub state read 2026-08-09
    revision = "b8a89d3096cf11a71d57b283c854f5ae2ed3df83"
    positive_label = "INJECTION"
