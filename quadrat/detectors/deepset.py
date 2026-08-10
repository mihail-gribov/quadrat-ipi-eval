#!/usr/bin/env python3
"""deepset's injection classifier, deberta-v3-base.

    python3 -m quadrat.run --detector deepset

Eighth on the indirect-injection leaderboard of `pi-detector-bench` (AUC 0.787) and one of the
oldest published detectors of this kind, which is why it belongs in the table: a reader comparing
2026 systems needs to see what the field looked like before them. Labels are `LEGIT` / `INJECTION`.

Its training set (`deepset/prompt-injections`) is public and small, and other entries here were
trained on it -- FMOPS distilbert among them. That makes deepset a shared ancestor of several rows
rather than an independent one, and it is worth remembering when two of them agree.
"""
from __future__ import annotations

from ..detector import register
from ._hf import _HFClassifier


@register("deepset", version="deberta-v3-base")
class Deepset(_HFClassifier):
    display = "deepset DeBERTa v3 base"
    model_id = "deepset/deberta-v3-base-injection"
    #: pinned: hub state read 2026-08-09
    revision = "80dda00d0b0d9a03917a7685e2ddbcd28e04dbb1"
    positive_label = "INJECTION"
