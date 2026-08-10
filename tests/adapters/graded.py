#!/usr/bin/env python3
"""A scored detector that needs no model, for exercising the thresholded path.

`floor` is binary, so a run of it never touches threshold selection, the curve or the operating
points -- which is most of what can go wrong. This one returns a continuous score, deterministically
derived from the document, so those paths run in a second and their invariants can be asserted.

It is deliberately IMPERFECT and deliberately NOT TIED: a detector that separated the classes
completely would make every threshold equivalent and hide exactly the bugs worth catching, and one
that returned the same value twice would sit on the score floor the runner refuses to publish under.
"""
from __future__ import annotations

import hashlib

from quadrat.detector import Detector, register


@register("graded", version="test")
class Graded(Detector):
    display = "Graded test detector"
    notes = "deterministic pseudo-score, no model"

    def score(self, docs):
        for d in docs:
            # A stable, distinct value per document: the id's hash spread over (0, 0.5).
            h = int(hashlib.sha1(d.id.encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
            base = h * 0.5
            # Injected documents are pushed up, but only by half the range, so the classes overlap
            # and recall actually depends on where the threshold falls.
            yield base + 0.5 if "audit@example.invalid" in d.text and h > 0.2 else base
