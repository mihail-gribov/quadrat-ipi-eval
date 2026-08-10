#!/usr/bin/env python3
"""Loading the corpus, and the slices worth measuring separately.

The dataset ships flat -- one JSONL of positives, one of negatives -- and the slices below are
filters over it, not separate files. They exist because several of the corpus's caveats are
one-field questions ("does the number hold without the leaked email?", "without the weak writer's
injections?"), and a caveat you can answer with a filter is a caveat the reader can check.
"""
from __future__ import annotations

import hashlib
import json
import pathlib
import typing as t

from . import pseudonym as pn

from .detector import Doc

from .paths import DATA_ROOT as DEFAULT_ROOT, corpus

#: name -> predicate on the raw row. Each is a caveat the reader can re-measure.
SLICES: dict[str, t.Callable[[dict], bool]] = {
    "all":         lambda r: True,
    "strong":      lambda r: r.get("gen_model") not in ("haiku", None),
    "obfuscated":  lambda r: bool(r.get("obfuscation")),
    "clean_typo":  lambda r: not r.get("typography_folded"),
    # NOT A RESEARCH SLICE -- a legal gate. `pii=true` marks the rows carrying real names of
    # living people (Enron, Clinton), and sending those to a third-party API is a transfer of
    # personal data: a separate act from publishing the corpus, covered by neither the research
    # framing nor a non-commercial tier. Any detector reached over someone else's API runs on
    # THIS slice, and the table says on which strata its row was obtained -- the composition is
    # narrower than everyone else's and that must not read as a quiet advantage.
    # The flag sits only on the mail rows; elsewhere the field is absent, so the predicate works
    # as written.
    "no_pii":      lambda r: not r.get("pii"),
}


def fingerprint(root=DEFAULT_ROOT):
    """Content hash of what a detector actually reads: `id`, `label`, `text`. Nothing else.

    Recorded with every result and checked when scores are reused. Document ids are positional
    (`pos-000123`), so a rebuilt dataset silently reuses them for different text -- reusing saved
    scores across a rebuild would then compute metrics for one corpus from another's scores, and
    nothing about the numbers would look wrong.

    IT USED TO HASH THE RAW BYTES of both files, which tied a measurement's identity to every
    metadata column sitting beside the text. Dropping one unused field then invalidated nine
    finished measurements that had read exactly the same documents. The question is "did the
    corpus a detector reads change", not "did anything in the file change".
    """
    root = corpus(root)
    h = hashlib.sha256()
    for name in ("positives.jsonl", "negatives.jsonl"):
        h.update(name.encode())
        with (pathlib.Path(root) / name).open(encoding="utf-8") as fh:
            for line in fh:
                r = json.loads(line)
                h.update(f"{r['id']}\0{r['label']}\0{r['text']}\0".encode())
    return h.hexdigest()[:16]


def meta_docs(root=DEFAULT_ROOT) -> dict[str, Doc]:
    """id -> Doc carrying every axis but NO text.

    Everything downstream of scoring -- curves, a second operating point, the whole per-cell
    profile -- is arithmetic over labels and axes; the text was only ever needed by the model.
    Dropping it turns a 380 MB load into a 100 MB one and makes re-deriving metrics from a saved
    scores file cheap enough to do on demand."""
    root = corpus(root)
    out = {}
    for name in ("positives.jsonl", "negatives.jsonl"):
        with (pathlib.Path(root) / name).open(encoding="utf-8") as fh:
            for line in fh:
                r = json.loads(line)
                r["text"] = ""
                out[r["id"]] = _doc(r)
    return out


def scored(scores_path, root=DEFAULT_ROOT, meta=None):
    """Re-read a saved `<run>.scores.jsonl` as (positives, negatives) of (doc, score).

    An id absent from the corpus means the scores belong to another build, and that is raised
    rather than skipped: silently dropping those ids would compute a clean-looking result from a
    partly foreign run."""
    meta = meta if meta is not None else meta_docs(root)
    pos, neg = [], []
    with pathlib.Path(scores_path).open(encoding="utf-8") as fh:
        for line in fh:
            r = json.loads(line)
            d = meta.get(r["id"])
            if d is None:
                raise KeyError(f"{r['id']} is not in the current build -- these scores belong to another corpus")
            (pos if d.label == "injected" else neg).append((d, r["score"]))
    return pos, neg


def _doc(r: dict) -> Doc:
    span = r.get("inj_span")
    return Doc(
        id=r["id"], text=r["text"], label=r["label"],
        host_type=r["host_type"], host_source=r["host_source"],
        family=r.get("family"), action=r.get("action"),
        spliced_at=r.get("spliced_at"),
        # flattened to the kind, so it can be a marginal axis: the classic evasion question is
        # "which distortion does it survive", not "was anything applied"
        obfuscation=((r.get("obfuscation") or {}).get("kind") or "none") if r.get("label") == "injected" else None,
        inj_span=tuple(span) if span else None,
        meta={k: r[k] for k in ("pii", "gen_model", "obfuscation", "inj_verified",
                                "typography_folded", "license")
              if k in r},
    )


def _replace_text(doc, text):
    """A copy of `doc` with different text. Dataclasses are frozen here on purpose -- a document
    that can be edited in place is a document whose id no longer describes its content."""
    import dataclasses
    return dataclasses.replace(doc, text=text)


def load(root=DEFAULT_ROOT, slice_name="all", limit=None, pseudonymise=False):
    """Return (positives, negatives) as lists of Doc, filtered by a named slice.

    The slice applies to BOTH sides: a leak-free measurement has to drop the leaked negatives too,
    or the FPR is computed on a pool the recall was not.

    `pseudonymise` rewrites the addresses in the CARRIER on the way out of this function -- for a
    run whose text leaves the machine. The injected span is left alone (its addresses are the
    payload, and they are already fake), and nothing is written to disk: the corpus is frozen and
    its fingerprint has to keep meaning what it means. See pseudonym.py."""
    root = corpus(root)
    keep = SLICES[slice_name]
    out = []
    for name in ("positives.jsonl", "negatives.jsonl"):
        rows = []
        with (root / name).open() as fh:
            for line in fh:
                r = json.loads(line)
                if keep(r):
                    d = _doc(r)
                    if pseudonymise:
                        d = _replace_text(d, pn.rewrite(d.text, d.inj_span))
                    rows.append(d)
                    if limit and len(rows) >= limit:
                        break
        out.append(rows)
    return out[0], out[1]
