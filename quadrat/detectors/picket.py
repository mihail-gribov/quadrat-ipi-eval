"""AI Cordon Picket -- ours, the signature half of the pair.

BINARY BY CONSTRUCTION, and declared so: a signature fires or it does not, so there is no score to
threshold and no curve to place an operating point on. `binary = True` routes the runner past
threshold selection to a fixed tau, and the reported FPR is the one Picket's own point produces
rather than one this harness chose for it. Without the flag, threshold selection on a verdict
vector degenerates -- see the note in `floor.py`.

The consequence for reading its row: Picket cannot be moved along a curve to match another
detector's FPR, so its recall and its FPR are a single package. Comparisons that hold FPR fixed do
not apply to it.

    pip install aicordon
"""
from __future__ import annotations

from ..detector import Detector, register


@register("picket", version="0.2.0")
class Picket(Detector):
    display = "AI Cordon Picket"
    binary = True
    notes = "signature/lexical, CPU"

    def setup(self):
        # The version in the decorator is what the result row will claim was measured, and it is
        # written by hand: the registry is read STATICALLY, by parsing this file, so the argument
        # has to stay a literal -- a named constant there parses as a Name and the scan yields
        # nothing. It is checked against the installed distribution here instead: a bumped package
        # under a stale decorator would publish one ruleset's numbers under another's name.
        from importlib.metadata import version as _dist_version
        got = _dist_version("aicordon")
        from ..detector import REGISTRY
        declared = REGISTRY["picket"][1]
        if got != declared:
            raise RuntimeError(f"picket: the adapter declares {declared}, {got} is installed -- "
                               f"fix version= in @register, or reinstall the package")
        from aicordon.picket import load
        self.det = load()

    def score(self, docs):
        # Picket is fast enough that a full pass is minutes, which is exactly why it printed
        # nothing -- and a silent process is indistinguishable from a hung one. Eighty thousand
        # documents at eighty a second is still a quarter of an hour of an operator watching a log
        # that never moves, so it reports like every other adapter.
        import time

        from aicordon.core.model import Document
        docs = list(docs)
        base = getattr(self, "progress_offset", 0)
        total = getattr(self, "progress_total", 0)
        if getattr(self, "_t0", None) is None:
            self._t0 = time.time()
        flagged = set()
        for f in self.det.scan(Document(id=d.id, text=d.text) for d in docs):
            flagged.add(f.doc_id)
        if total:
            el = time.time() - self._t0
            seen = base + len(docs)
            rate = seen / el if el else 0
            left = (total - seen) / rate / 60 if rate else 0
            print(f"  Picket · documents {seen}/{total} · {rate:.0f} docs/s · "
                  f"~{max(0, left):.0f} min left", flush=True)
        for d in docs:
            yield 1.0 if d.id in flagged else 0.0
