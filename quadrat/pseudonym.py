#!/usr/bin/env python3
"""Deterministic pseudonymisation of the addresses in a carrier -- for text leaving the machine.

Not required by anything here: the mail carriers are already-public records (FERC, FOIA), so
sending them to an API discloses nothing new. This exists so a run over someone else's service can
be made harmless without touching the corpus, and it is applied on the way out, never on disk.

WHAT IT DOES, and why in this shape:

    jeff.skilling@enron.com  ->  jeff.skilling+k3f9@example.invalid

* **The name survives the @.** Mail bodies refer to people by name and then address them; blanking
  the local part would break that agreement and hand the detector text no correspondence ever
  looked like. The suffix is what blurs it -- the string is no longer the person's real address
  while still reading as theirs in context.
* **The domain goes.** `enron.com` is the organisation, and it is the part that makes the address
  resolvable and attributable. `.invalid` is reserved by RFC 2606 and can never exist.
* **ONE MAP FOR THE WHOLE CORPUS.** The same address must map to the same replacement everywhere,
  or a thread that quotes a colleague twice becomes a thread with two colleagues -- and a detector
  reading a mail for internal consistency would see a document no mail system produces. The map is
  derived from a hash rather than assigned in order of appearance, so it does not depend on which
  documents were loaded, in what order, or on how many.

WHAT IT DOES NOT TOUCH: the injected span. Its addresses ARE the payload -- an exfiltration target
is the thing the detector is supposed to notice -- and they are already fake, sitting under
RFC-reserved names by construction (0 live domains out of 1417). Rewriting them would blur the
signal rather than the person.

The map can be dumped for audit; nothing about it is secret, and the point is that a reader can
check what left the machine.
"""
from __future__ import annotations

import hashlib
import re

#: NO \b ANCHORS, and that is the whole point of this comment. They were here first, as
#: "conservative" -- and they made the pattern miss precisely the messy cases that matter:
#:
#:     pilar.ramirez@enron.com________   trailing `_` is a word character, so \b never matches
#:     veprekam@state.gov02-647-1512    address run together with a phone number
#:     HDR22@clintonemail.com1          OCR/---export debris glued to the domain
#:
#: Eight real addresses survived the first build for exactly this reason, and every one of them
#: was a genuine name at a genuine domain. The character classes already stop where they should:
#: `[A-Za-z]{2,}` cannot eat the `0` of `gov02`, so the TLD ends on its own. Anchoring the end
#: only added a way to fail.
#:
#: The lesson generalises to any redaction: verify by counting what is LEFT, never by counting
#: what was changed. A pass that rewrote 13252 documents looked like a success.
EMAIL = re.compile(r"([A-Za-z0-9._%+-]+)@([A-Za-z0-9.-]+\.[A-Za-z]{2,})")

#: RFC 2606 reserves this TLD: it cannot resolve, and it reads as synthetic at a glance.
DOMAIN = "example.invalid"

#: The separator is `+`, RFC 5233 subaddressing -- the conventional way to add a tag to a local
#: part, and unmistakably part of it. A `.` was used first and it read as a domain label: the
#: string `enronXgate.ca7f@example.invalid` fooled this file's own verification into reporting a
#: surviving `enronXgate.ca`. If a check can be misread that way, so can a reader.
SEP = "+"

#: How much of the hash goes into the local part. Four hex characters is 65536 values -- enough
#: that two different addresses sharing a local part do not collide in a corpus of this size, and
#: short enough that the name still dominates what a reader sees.
SUFFIX_LEN = 4


class Table:
    """address -> replacement, stable by construction and inspectable afterwards."""

    def __init__(self, salt: str = ""):
        self.salt = salt
        self.map: dict[str, str] = {}

    def replacement(self, local: str, domain: str) -> str:
        addr = f"{local}@{domain}".lower()
        got = self.map.get(addr)
        if got is None:
            h = hashlib.sha256((self.salt + addr).encode()).hexdigest()[:SUFFIX_LEN]
            got = f"{local}{SEP}{h}@{DOMAIN}"
            self.map[addr] = got
        return got

    def rewrite_span(self, text: str, keep: tuple[int, int] | None = None):
        """(new_text, new_keep). Rewrite every address outside `keep`, and MOVE `keep` with it.

        A replacement is not the same length as what it replaces, so every address rewritten
        before the injection shifts it. Returning the text alone was wrong in a way that looks
        right: the corpus still carried the old offsets, `inj_span` pointed a few characters off,
        and the very span this function exists to protect no longer described the injection. The
        guard is measurable -- `check_invariants` compares the span's text between builds."""
        a, b = keep if keep else (-1, -1)
        out, last, shift, moved = [], 0, 0, keep
        for m in EMAIL.finditer(text):
            if a <= m.start() and m.end() <= b:      # inside the injection: leave it alone
                continue
            rep = self.replacement(m.group(1), m.group(2))
            out.append(text[last:m.start()])
            out.append(rep)
            last = m.end()
            if keep and m.end() <= a:                # entirely before the span: it moves
                shift += len(rep) - (m.end() - m.start())
        out.append(text[last:])
        if keep:
            moved = (a + shift, b + shift)
        return "".join(out), moved

    def rewrite(self, text: str, keep: tuple[int, int] | None = None) -> str:
        """Text only, for callers that do not carry the span onward."""
        return self.rewrite_span(text, keep)[0]

    def dump(self, path):
        import json
        import pathlib
        p = pathlib.Path(path)
        p.write_text(json.dumps(self.map, ensure_ascii=False, indent=1), encoding="utf-8")
        return p


#: One table per process, so every adapter and every document share the same mapping.
TABLE = Table()


def rewrite(text, keep=None):
    return TABLE.rewrite(text, keep)
