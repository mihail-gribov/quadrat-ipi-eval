#!/usr/bin/env python3
"""Bastion Prompt Protection (bastionsoft), deberta-v3-xsmall.

    python3 -m quadrat.run --detector bastion

The detector that tops the indirect-injection leaderboard of `pi-detector-bench` (AUC 0.952 at
70M parameters, ahead of Wolf Defender 0.865 and ProtectAI v2 0.816). Worth measuring here
precisely because that number comes from a benchmark its own authors maintain: the leaderboard's
README already warns that BIPIA cannot be used head-to-head because it trained PIGuard, and the
same question applies to a maintainer scoring first on their own board. This corpus answers it --
the positives are freshly generated and none of them existed when this checkpoint was published
(2026-06-16, pinned below).

Small: 12 layers at hidden 384, 512 positions. The window is therefore the usual one and is not
optional -- 93% of the doc carrier is longer than 512 tokens, so an unwindowed pass would measure
truncation and report it as detection.

Its labels are `safe` / `attack` rather than the BENIGN/INJECTION convention, which is why the
positive class is named here and resolved against the checkpoint's own id2label: a model that
orders its classes differently must not be able to silently invert the metric.

AGPL-3.0. That governs the model, not our measurement of it; the row reports published weights at
their default configuration, as every other row does.
"""
from __future__ import annotations

from ..detector import register
from ._hf import _HFClassifier


@register("bastion", version="deberta-v3-xsmall-v1")
class Bastion(_HFClassifier):
    display = "Bastion Prompt Protection"
    model_id = "bastionsoft/binary-bastion-prompt-protection-deberta-v3-xsmall-v1"
    #: pinned: hub state of 2026-06-16. `main` moves, and a measured row has to name what it
    #: measured -- especially for a detector whose authors also keep the leaderboard.
    revision = "3a5bbe0e8eadf86213378e4806da42a1a3177df8"
    positive_label = "ATTACK"
    fallback_index = 1
