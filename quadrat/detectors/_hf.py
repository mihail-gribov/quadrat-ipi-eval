#!/usr/bin/env python3
"""Shared base for the published transformer classifiers -- not a detector itself.

Leading underscore on purpose: the runner auto-imports every module in this directory, and this
one registers nothing. Each detector gets its own file so its adapter can be replaced, version
-bumped or contributed by its author without touching a file its competitors live in -- an adapter
encodes threshold, preprocessing and checkpoint, and those choices belong to whoever ships it
(adapters/README.md).

The classifiers here are DeBERTa-family sequence models with a 512-token limit, differing only in
checkpoint and in which logit means "injection", so the mechanism lives once, here. They are
scored rather than binary, which is what lets the runner put every system on the same false
positive rate instead of comparing default cut-offs chosen against different data.

    pip install torch transformers

THE WINDOW IS DECLARED, NOT IMPOSED. These models stop at 512 tokens (~2000 characters) and 93% of
the doc carrier is longer, so handed a whole document they would silently read its head and the
tail would never reach the model -- which measures truncation and reports it as detection. So the
limit is declared here as `max_chars`, and the base class splits on sentence boundaries with a
four-sentence overlap and folds the windows back by max. That is what a deployment of a 512-token
classifier does, and it is the setting under which the number is about the model rather than about
whoever integrated it.

Measuring the naive integration instead is still possible and still interesting -- `--policy
truncate` on the runner overrides the declaration -- but it is an experiment about integrations,
not the detector's row in the table.

The models are the checkpoints these projects publish, at their default configuration. We do not
tune them: an adapter encodes choices, and choices about someone else's detector belong to its
author (see adapters/README.md).
"""
from __future__ import annotations

import os
import time

from ..detector import Detector
from ..window import WINDOW_512

#: Defaults for a 512-token DeBERTa. BOTH are per-adapter, because they are not properties of
#: this file: a model with an 8192-token window truncated at 512 would be handed the aperture it
#: declared and shown a fraction of it -- the declaration would be a lie the numbers cannot show.
#: And a batch sized for short sequences will not fit a long-context model on the same card.
BATCH = int(os.environ.get("QUADRAT_HF_BATCH", "32"))
MAXLEN = 512
#: documents between progress lines. A classifier pass is tens of minutes on this corpus, and a
#: log that says nothing until it finishes cannot be told from one that has hung.
PROGRESS_EVERY = int(os.environ.get("QUADRAT_HF_PROGRESS", "2000"))


class _HFClassifier(Detector):
    """One transformer, scored by the probability of its injection class."""

    #: documents in the batch currently being scored -- set by the runner's loop through
    #: `score_documents`, and used only to turn a window rate into a document estimate
    _docs_in_batch: int = 0

    #: 512 tokens, in the characters the harness measures in. The base class does the splitting.
    max_chars = WINDOW_512
    model_id: str = ""
    #: hub revision. Pin it: `main` moves, and a measured row has to name what it measured.
    revision: str | None = None
    #: label whose probability is the score. Resolved against the checkpoint's own id2label, so a
    #: model that orders its classes differently cannot silently invert the metric.
    positive_label: str = "INJECTION"
    fallback_index: int = 1
    #: For a MULTI-CLASS head. When set, the score is the summed probability of these labels --
    #: a detector that splits attacks into kinds still answers one question here ("is this an
    #: attack"), and its taxonomy is its own. Picking a single class instead would score a
    #: disagreement about naming as a miss.
    positive_labels: tuple[str, ...] = ()
    #: tokens per forward. MUST match the declared aperture: see the note at the top.
    max_tokens: int = MAXLEN
    #: sequences per forward. Lower it for a long-context model sharing a card.
    batch: int = BATCH

    def _load(self):
        """(model, tokenizer) for this checkpoint, at its stock configuration.

        A hook, not just a step: a checkpoint that ships a custom class overrides this instead of
        re-implementing batching and scoring. `trust_remote_code` stays off everywhere -- an
        adapter that needs vendor code re-states those lines locally, where they can be read and
        pinned, rather than executing whatever the hub branch holds today."""
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        model = AutoModelForSequenceClassification.from_pretrained(
            self.model_id, revision=self.revision, trust_remote_code=False)
        tok = AutoTokenizer.from_pretrained(self.model_id, revision=self.revision,
                                            trust_remote_code=False)
        return model, tok

    def setup(self):
        import torch

        self.torch = torch
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model, self.tok = self._load()
        self.model = self.model.to(self.device).eval()
        id2label = getattr(self.model.config, "id2label", {}) or {}
        if self.positive_labels:
            want = {w.upper() for w in self.positive_labels}
            self.indices = sorted(i for i, lab in id2label.items()
                                  if str(lab).upper() in want)
            missing = want - {str(lab).upper() for lab in id2label.values()}
            if missing or not self.indices:
                raise SystemExit(f"[{self.model_id}] the checkpoint has no classes {sorted(missing)}; "
                                 f"it has {sorted(str(v) for v in id2label.values())}")
            self.index = self.indices[0]
        else:
            self.index = next((i for i, lab in id2label.items()
                               if str(lab).upper().startswith(self.positive_label[:4])),
                              self.fallback_index)
            self.indices = [self.index]
        rev = f"@{self.revision[:8]}" if self.revision else ""
        pos = ("+".join(str(id2label.get(i)) for i in self.indices)
               if len(self.indices) > 1 else str(id2label.get(self.index)))
        self.notes = f"{self.model_id}{rev} · {self.device} · positive={pos}"

    def score_documents(self, docs):
        docs = list(docs)
        self._docs_in_batch = len(docs)
        return super().score_documents(docs)

    def score(self, docs):
        # `docs` here are WINDOWS, not documents -- the base class splits before scoring -- so the
        # count reported is windows, and the document figure comes from the runner's own offsets.
        base = getattr(self, "progress_offset", 0)
        total = getattr(self, "progress_total", 0)
        # `_t0` spans the whole pass, `n` counts windows WITHIN one 200-document batch. Dividing
        # the second by the first gave a rate that fell towards zero as the run went on and an ETA
        # that grew with every line -- 6759 minutes left on a pass with forty minutes to go. So the
        # windows finished by earlier batches are carried in `_win_base` and the rate is computed
        # over the pass, which is the only pair of numbers taken in the same units.
        if getattr(self, "_t0", None) is None:
            self._t0 = time.time()
            self._win_base = 0
        n = 0
        batch = []
        for d in docs:
            batch.append(d.text)
            if len(batch) == self.batch:
                yield from self._run(batch)
                n += len(batch)
                batch = []
                self._tick(base, total, n)
        if batch:
            yield from self._run(batch)
            n += len(batch)
            self._tick(base, total, n, last=True)
        self._win_base += n

    def _tick(self, base, total, n, last=False):
        """`n` counts WINDOWS, `base`/`total` count DOCUMENTS -- they are different units and were
        printed as one, so with four windows per document the counter ran ahead of the corpus and
        then sat clamped at its size. Windows are what this loop can honestly count, so windows
        are what it reports, with the document position beside them."""
        if not total:
            return
        if not last and n % PROGRESS_EVERY >= self.batch:
            return
        el = time.time() - self._t0
        wins = self._win_base + n
        rate = wins / el if el else 0
        # Windows per document, measured on the batches that FINISHED -- `base` counts exactly
        # those documents, so this ratio is a division of two completed quantities rather than an
        # estimate. Before the first batch lands there is nothing to divide, and the line says so
        # instead of printing a number derived from one partial batch.
        per_doc = (self._win_base / base) if base else 0
        if per_doc and rate:
            seen = base + min(self._docs_in_batch, n / per_doc)
            left = (total - seen) * per_doc / rate / 60
            tail = f"{rate:.0f} windows/s · ~{max(0, left):.0f} min left"
        else:
            tail = f"{rate:.0f} windows/s · -- left (first batch)"
        print(f"  {self.model_id.split('/')[-1]} · documents {base}/{total} · "
              f"{n} windows in batch, {tail}",
              flush=True)

    def _run(self, texts):
        torch = self.torch
        enc = self.tok(texts, return_tensors="pt", truncation=True,
                       max_length=self.max_tokens, padding=True).to(self.device)
        with torch.no_grad():
            p = self.model(**enc).logits.softmax(-1)
            probs = p[:, self.indices].sum(-1) if len(self.indices) > 1 else p[:, self.index]
        return probs.float().cpu().tolist()

    def teardown(self):
        self.model = None
        if self.device == "cuda":
            self.torch.cuda.empty_cache()      # three of these run in one session; free the card


