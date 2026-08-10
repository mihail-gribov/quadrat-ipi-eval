#!/usr/bin/env python3
"""Windowing: how a detector with a bounded context sees a document that does not fit.

Not an implementation detail -- a declared parameter of the run, recorded beside the number.
Most detectors here cannot read a whole document (ProtectAI and PromptGuard stop at 512 tokens,
and 93% of the doc stratum is longer than that), and what happens to the remainder decides the
result:

    truncate   whatever fits, and the rest is gone. On this corpus a 2000-character head loses
               62% of the `end` injections, 35% of `sentence` and 23% of `paragraph` -- the
               detector never sees them, so the "recall" is largely reporting where we put the
               payload. Worth measuring, because it is what a naive integration does, but it is a
               property of the integration, not of the detector.
    sentence   one sentence per window. The finest resolution, and the setting under which a
               sentence-level detector is built; costs one call per sentence.
    chunk      sentences packed to the window, each chunk overlapping the last by two sentences.
               What a serious deployment does. The overlap is the point: an injection sitting on a
               chunk boundary is whole inside at least one window, so a miss is the detector's and
               not the splitter's.

Everything is cut on sentence boundaries (the same segmenter as the splicer). A window cut at a
character count hands the detector a fragment no deployment would produce, and then measures its
reaction to the fragment.

    from quadrat.window import windows
    for text, offset in windows(doc.text, size=2000, policy="chunk"): ...
"""
from __future__ import annotations

import re

_SENT_FALLBACK = re.compile(r"(?<=[.!?])[ \t]+(?=[A-Z\"'(])")

POLICIES = ("full", "truncate", "sentence", "chunk")

#: Overlap between consecutive windows. An integer is a number of SENTENCES; a value <= 1 is read
#: as a share of the window in characters. Default: four sentences.
#:
#: The purpose is narrow -- an injection landing on a boundary should sit whole inside at least
#: one window, so that a miss belongs to the detector and not to the splitter. Measured on the v1
#: core (16 800 injections, 2000-character window):
#:
#:     overlap          windows   injections torn
#:     2 sentences       57 796      392   2.33%
#:     3 sentences       64 480      263   1.57%
#:     4 sentences       72 633      167   0.99%
#:     6 sentences       92 735      114   0.68%
#:     8 sentences      117 830       98   0.58%
#:     25% of window     49 941      318   1.89%
#:     50% of window     58 739      122   0.73%
#:
#: A share is the stronger construction -- stepping by half a window *guarantees* every span
#: shorter than half a window is whole somewhere, while "four sentences back" is anywhere from 40
#: to 900 characters and guarantees nothing -- and at 50% it measured better on both axes than
#: four sentences (0.73% vs 0.99% torn, 19% fewer windows). Four sentences is nonetheless the
#: default: it is the unit a reader of the protocol can picture, and it matches what deployments
#: actually configure. The share form stays available for anyone who wants the guarantee.
#:
#: Residual tears are reported, not hidden. An injection longer than the window cannot be rescued
#: by any splitter; that one is the detector's limit.
OVERLAP_SENTENCES = 4

#: The convention here: 512 tokens = 2000 characters.
#:
#: Windows are declared in CHARACTERS, not tokens, because the RUNNER has no business knowing any
#: detector's tokenizer -- each detector knows its own perfectly well, but the harness cuts the
#: text before any of them sees it, and it must cut identically for all. So the aperture is agreed
#: in characters: at roughly 4 characters per token on English prose, 512 tokens (the limit of
#: ProtectAI, PromptGuard and the DeBERTa family) lands near 2000. Every detector is then measured
#: through the same opening, whatever it tokenises it into, and comparing two of them is not
#: comparing their tokenizers.
WINDOW_512 = 2000


def sentence_spans(text):
    """[(start, end)] over the whole text, contiguous, so joining them rebuilds the document."""
    try:
        import blingfire
        _, offs = blingfire.text_to_sentences_and_offsets(text)
        ends = [b for a, b in offs if text[a:b].strip()]
    except Exception:
        ends = [m.start() for m in _SENT_FALLBACK.finditer(text)]
    ends = sorted(e for e in ends if 0 < e < len(text))
    spans, prev = [], 0
    for e in ends + [len(text)]:
        if e > prev:
            spans.append((prev, e))
            prev = e
    if spans and spans[-1][1] < len(text):        # trailing whitespace belongs to the last span,
        spans[-1] = (spans[-1][0], len(text))     # otherwise the windows miss the final character
    return spans or [(0, len(text))]


def _capped(chunk, start, size):
    """Yield (text, offset) pieces of at most `size` characters.

    THE APERTURE IS A HARD BOUND, and one case can break it: a "sentence" longer than the whole
    window. Real text produces them -- a minified script, a base64 blob, a table with no sentence
    punctuation in it -- and there is no boundary inside to cut on, so the only choices are to emit
    a window larger than the declared aperture or to cut on characters. It has to be the cut.

    An oversized window is not a harmless overshoot that the detector sorts out. A local model
    tokenises and truncates it: the tail is silently dropped, and an injection there is scored as
    absent while the run reports full coverage. A hosted one refuses the request outright -- 413,
    payload too large -- and the pass dies mid-corpus. Both come from the harness handing over more
    than it said it would, so the harness is where it is fixed."""
    if len(chunk) <= size:
        yield chunk, start
        return
    for a in range(0, len(chunk), size):
        piece = chunk[a:a + size]
        if piece.strip():
            yield piece, start + a


def windows(text, size=None, policy="full", overlap=OVERLAP_SENTENCES):
    """Yield (chunk_text, offset). `offset` is the chunk's start in the document, so a detector
    that reports a span can have it mapped back.

    A document that already fits yields exactly one window under every policy -- the policies
    differ only once the text does not fit."""
    if policy == "full" or not size or len(text) <= size:
        yield text, 0
        return

    spans = sentence_spans(text)

    if policy == "sentence":
        for a, b in spans:
            if text[a:b].strip():
                yield from _capped(text[a:b], a, size)
        return

    if policy == "truncate":
        stop = 0
        for a, b in spans:
            if b > size and stop:                 # keep whole sentences only
                break
            stop = b
        yield text[:min(stop or size, size)], 0   # ...unless the first one alone overruns
        return

    # chunk: pack sentences up to `size`, then start the next window `overlap` back -- in
    # characters, snapped forward to a sentence start so no window opens mid-sentence.
    by_share = overlap <= 1
    step = max(size - int(size * overlap), size // 4) if by_share else 0
    i, n = 0, len(spans)
    while i < n:
        j, start = i, spans[i][0]
        while j < n and spans[j][1] - start <= size:
            j += 1
        j = max(j, i + 1)                          # a single oversized sentence still goes out,
        chunk = text[start:spans[j - 1][1]]        # but `_capped` cuts it down to the aperture
        # WHITESPACE-ONLY WINDOWS ARE NEVER SCORED. A trailing blank tail is not evidence of
        # anything, and handing it to a detector would spend a call to ask whether a space is an
        # injection -- and, for a scored detector, put a meaningless value into the threshold's
        # negative pool. Documents whose tail is only whitespace therefore show one uncovered
        # character in a coverage audit; that is the rule working, not a gap.
        if chunk.strip():
            yield from _capped(chunk, start, size)
        if j >= n:
            break
        if by_share:
            nxt, k = start + step, i + 1
            while k < j and spans[k][0] < nxt:     # first sentence at or past the step
                k += 1
            i = min(k, j)
        else:
            i = max(j - int(overlap), i + 1)       # step back N sentences, but always progress


def score_windowed(detector, docs, size, policy, overlap=OVERLAP_SENTENCES, aggregate=max):
    """Score every window, fold back to one score per document, return (scores, n_windows).

    `max` is the aggregate: a document is as suspicious as its most suspicious part. Averaging
    would dilute a real injection into a long clean carrier, turning the score into a length
    artefact -- longer documents would look safer for no reason."""
    parts, owner = [], []
    for i, d in enumerate(docs):
        for chunk, _off in windows(d.text, size=size, policy=policy, overlap=overlap):
            parts.append(_Chunk(d, chunk))
            owner.append(i)

    part_scores = list(detector.score(parts))
    if len(part_scores) != len(parts):
        raise ValueError(f"detector returned {len(part_scores)} scores for {len(parts)} windows")

    out = [None] * len(docs)
    for idx, s in zip(owner, part_scores):
        out[idx] = s if out[idx] is None else aggregate(out[idx], s)
    return [0.0 if s is None else s for s in out], len(parts)


class _Chunk:
    """A document view over one window: same metadata, replaced text."""

    __slots__ = ("_d", "text")

    def __init__(self, doc, text):
        self._d = doc
        self.text = text

    def __getattr__(self, name):
        return getattr(self._d, name)
