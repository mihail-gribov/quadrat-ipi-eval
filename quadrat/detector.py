#!/usr/bin/env python3
"""The one thing you implement to measure a detector.

A detector is anything that turns a document into a score. Subclass `Detector`, implement `score`,
register it, and `run.py` does the rest: choosing the threshold, the per-cell profile, the
confidence intervals, and the results row.

    from quadrat.detector import Detector, register

    @register("my-detector", version="1.2")
    class MyDetector(Detector):
        def setup(self):                       # optional: load weights, warm a connection
            self.model = load_model()

        def score(self, docs):                 # REQUIRED: yield one float per document, in order
            for d in docs:
                yield self.model.predict(d.text)

THE APERTURE IS YOURS. The harness hands `score` whole documents. If your detector has a
context limit, declare it as `max_chars` and the base class splits for you (sentence-aligned
windows, four-sentence overlap, folded back by max) -- or override `score_documents` and split
however your product does. What it will never do is impose a window on a detector that did not
ask for one: how to read a long document is part of the detector, and a harness that chunked for
everyone would be reporting its own splitter's recall. `picket` and `floor` declare no limit and
so are measured on whole documents.

WHAT `score` MUST RETURN. A continuous score where higher = more likely an injection. Continuity
matters: the whole protocol sets a threshold at a target FPR, and a binary verdict cannot be
thresholded -- averaging a binary verdict yields (recall + specificity) / 2, which looks like an
AUC and is not one. If your detector is genuinely binary (a signature fires or it does not), say
so with `binary = True` and the runner will report it at its own operating point instead of
pretending a curve exists.

BATCHING. `score` receives an iterable of documents, not one document, so a detector that batches
(GPU, or an API that takes many texts per call) can do so without fighting the interface. Yield in
input order; the runner zips scores back to documents by position.

Docs: README.md, section "Adding a detector"
"""
from __future__ import annotations

import dataclasses
import typing as t

from .window import OVERLAP_SENTENCES


@dataclasses.dataclass(frozen=True)
class Doc:
    """One corpus document. `label` and the cell fields are present for positives only."""
    id: str
    text: str
    label: str                       # "injected" | "clean"
    host_type: str                   # email | doc | web
    host_source: str                 # enron | clinton | billsum | cnndm | govreport | fineweb
    family: str | None = None        # cell axis 1 -- the lever
    action: str | None = None        # cell axis 2 -- what happens on success
    spliced_at: str | None = None    # end | paragraph | sentence -- where in the carrier it sits
    obfuscation: str | None = None   # homoglyph | zero_width | leetspeak | none
    inj_span: tuple[int, int] | None = None
    meta: dict = dataclasses.field(default_factory=dict)

    @property
    def cell(self) -> str | None:
        return f"{self.family}/{self.action}" if self.family else None


class Detector:
    """Base class. Implement `score`; everything else is optional."""

    #: set True if the detector emits a verdict rather than a score (see the module docstring)
    binary: bool = False
    #: Largest document this detector will accept, in characters (None = no limit).
    #:
    #: THIS IS WHAT DECIDES THE APERTURE. The harness feeds whole documents; a detector that
    #: cannot read one is responsible for its own splitting, because how to split is part of the
    #: detector, not of the measurement. A harness that chunked for everyone would be measuring
    #: its own splitter, and would hand a bounded model the aperture the harness guessed rather
    #: than the one its author would ship.
    #:
    #: Leaving it None therefore means "give me the document" -- and `picket` and `floor` do,
    #: because a signature or a regex has no context limit to respect.
    max_chars: int | None = None
    #: How this detector reads a document longer than `max_chars`. Consulted only when a limit is
    #: declared; see `window.POLICIES`. `chunk` is the default because it is what a deployment
    #: does, and because the overlap keeps a boundary-straddling injection whole in some window.
    policy: str = "chunk"
    #: Sentences of overlap between its windows. Same condition: only with a limit.
    overlap: int = OVERLAP_SENTENCES
    #: free text recorded alongside the results row
    notes: str = ""
    #: How the system is called by its authors -- "Bastion Prompt Protection", not "bastion".
    #: The registry name is an identifier for the command line; a report heading is prose, and a
    #: vendor's product deserves its own name in it. Empty falls back to the registry name.
    display: str = ""

    def aperture(self) -> tuple[str, int | None, int | None]:
        """(policy, size, overlap) this detector reads through. Recorded with every result.

        Comes from the detector's own declaration, never from a flag: two rows are comparable
        because each system was measured the way it actually reads, not because both were forced
        through the same window."""
        if not self.max_chars:
            return ("full", None, None)
        return (self.policy, self.max_chars,
                self.overlap if self.policy == "chunk" else None)

    def score_documents(self, docs) -> tuple[list[float], int]:
        """What the runner calls. Applies the detector's own aperture, then scores.

        Returns (scores, windows_seen). An adapter that wants different internal handling --
        its own segmenter, a two-stage pass, a cache -- overrides this and keeps everything
        below it."""
        policy, size, overlap = self.aperture()
        if policy == "full":
            docs = list(docs)
            return list(self.score(docs)), len(docs)
        from .window import score_windowed
        return score_windowed(self, docs, size, policy, overlap or OVERLAP_SENTENCES)

    #: Set by the runner before each batch: how many documents of this pass are already done, and
    #: how many there are in total. An adapter printing progress reports the PASS, not its slice.
    progress_offset: int = 0
    progress_total: int = 0

    def setup(self) -> None:
        """Load models, open connections. Called once before scoring."""

    def score(self, docs: t.Iterable[Doc]) -> t.Iterable[float]:
        """Yield one score per document, in input order. Higher = more injection-like."""
        raise NotImplementedError

    def teardown(self) -> None:
        """Release anything setup() acquired."""


REGISTRY: dict[str, tuple[type[Detector], str]] = {}


def register(name: str, *, version: str):
    """Register a detector under a name the runner can select with --detector."""
    def deco(cls):
        if not issubclass(cls, Detector):
            raise TypeError(f"{cls.__name__} must subclass Detector")
        REGISTRY[name] = (cls, version)
        return cls
    return deco


def load_detectors(extra_dirs=()) -> None:
    """Import the built-in reference detector, then any adapter directories given.

    `quadrat/detectors/` holds the adapters the published table was measured with. `extra_dirs`
    loads more from anywhere -- that is how you add your own without forking, and how the tests
    supply one that needs no model.

    AN ADAPTER IS A CLAIM, not a wrapper: it encodes the threshold, the preprocessing and the
    model version, and whoever writes it owns those choices. Ours are here on the same terms as
    everyone else's, and none of them is a dependency of the harness itself -- the only detector
    that runs out of the box is `floor`, five regexes that import nothing."""
    import importlib
    import importlib.util
    import pathlib
    import pkgutil
    pkg = pathlib.Path(__file__).resolve().parent / "detectors"
    for m in pkgutil.iter_modules([str(pkg)]):
        if not m.name.startswith("_"):
            importlib.import_module(f"quadrat.detectors.{m.name}")

    for d in extra_dirs:
        d = pathlib.Path(d).resolve()
        for f in sorted(d.glob("*.py")):
            if f.name.startswith("_"):
                continue
            spec = importlib.util.spec_from_file_location(f"quadrat_adapter_{f.stem}", f)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
