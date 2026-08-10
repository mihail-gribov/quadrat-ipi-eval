"""The triviality floor: five regexes for the most quoted injection phrases.

WHAT IT IS FOR. Nobody ships this, and it is not competing. It exists so that every number in
every report has something to be read against. "Recall 12%" means nothing on its own; "recall 12%
where five regexes already get 6.5%" means the detector is worth about twice a grep. Two things
fall out of it that the report could not otherwise state:

  * a lower bound on the corpus. If a detector scores below the floor, the finding is not that it
    ranks last -- it is that it is not detecting anything, and the report should say so plainly.
  * a reading of the corpus itself. The floor's own recall is roughly the share of this corpus
    that is quotable boilerplate. That it is small is the point: the injections here are mostly
    not the phrases everyone tests against.

Every report therefore carries a floor column beside the measured detector, filled from a floor
pass over the SAME dataset and the SAME aperture -- a floor measured on another corpus or through
another window would make the comparison meaningless, so the runner refuses to borrow one.

Called `floor`, not `baseline`: every system measured here is a baseline in the paper's sense, so
that word cannot also name this one.

BINARY, and declared so. `binary = True` is not decoration: the runner picks a threshold at a
target FPR for scored detectors, and on a 0/1 vector that procedure silently degenerates -- with
fewer firings than the FPR budget it selects tau = 0 and `score >= tau` flags the entire corpus,
which reads as 100% recall at 100% FPR and looks like a result. The flag routes this detector past
threshold selection to a fixed tau = 0.5.

The file is not named `keyword.py`, tempting as that is: a module called `keyword` shadows the
standard library module of that name for any process with this directory on its path, and
`collections` imports it -- which breaks tools long before they reach anything of ours.
"""
from __future__ import annotations

import re

from ..detector import Detector, register

RX = re.compile(r"(?i)\b(ignore (all )?(previous|prior) instructions|disregard the above|"
                r"you are now|system prompt|as an AI language model)\b")


@register("floor", version="1.0")
class Floor(Detector):
    display = "Regex floor (triviality baseline)"
    binary = True
    notes = "5 regexes — the triviality floor"

    def score(self, docs):
        for d in docs:
            yield 1.0 if RX.search(d.text) else 0.0
