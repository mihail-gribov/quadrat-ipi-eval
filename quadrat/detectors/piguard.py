#!/usr/bin/env python3
"""PIGuard (ACL 2025), leolee99/PIGuard.

    python3 -m quadrat.run --detector piguard

PIGuard publishes its training set, which overlaps material we have measured it on elsewhere.
That overlap does not reach this dataset -- the injections here were written for it and did not
exist before -- but the carriers are public corpora, so the note stays.

THE CHECKPOINT DOES NOT LOAD WITH THE STOCK CLASS, and it fails silently rather than loudly. The
repo ships `modeling_piguard.py` behind the config's `auto_map`, normally reached with
`trust_remote_code=True`. Those lines do one thing that matters: the CLS vector of
`last_hidden_state` goes straight into `Linear(hidden, 2)`, BYPASSING the `ContextPooler` that
stock `DebertaV2ForSequenceClassification` places before its classifier. Load the checkpoint with
the stock class and the pooler is untrained -- the model still runs, still returns probabilities,
and scores noise. So the eight lines that matter are re-stated here, verified against the vendor
loader to ~1e-5 by `experiments/38_piguard_eval/smoke.py`, and `trust_remote_code` stays off: the
vendor file targets transformers 4.44 while this runs 5.x, and pinning a revision means nothing if
the code beside it is fetched from a moving branch.
"""
from __future__ import annotations

from ..detector import register
from ._hf import _HFClassifier

#: pinned: hub state of 2025-08-03, the revision this row was measured on
REVISION = "dd78b24e330193a22d2293ac66922dd4f982f563"


@register("piguard", version="acl2025")
class PIGuard(_HFClassifier):
    display = "PIGuard (ACL 2025)"
    model_id = "leolee99/PIGuard"
    revision = REVISION
    positive_label = "INJECTION"

    def _load(self):
        from transformers import AutoTokenizer, DebertaV2Config, DebertaV2ForSequenceClassification
        from transformers.modeling_outputs import SequenceClassifierOutput

        class PIGuardLocal(DebertaV2ForSequenceClassification):
            """The vendor forward: encoder, then CLS straight into the classifier.

            Only `forward` is overridden. `__init__` is left alone because
            `pooler_hidden_size == hidden_size == 768` here, so the inherited `classifier` already
            has the shape the checkpoint stores."""

            def forward(self, input_ids=None, attention_mask=None, **kwargs):
                out = self.deberta(input_ids=input_ids, attention_mask=attention_mask,
                                   output_hidden_states=False)
                return SequenceClassifierOutput(logits=self.classifier(out.last_hidden_state[:, 0, :]))

        # The config declares `model_type: piguard`, which no stock AutoConfig knows, so it is read
        # as the DebertaV2Config it actually is. Every field -- hidden sizes, id2label, the
        # 512-token window -- comes from the vendor file unchanged.
        cfg = DebertaV2Config.from_pretrained(self.model_id, revision=self.revision,
                                              trust_remote_code=False)
        model = PIGuardLocal.from_pretrained(self.model_id, revision=self.revision, config=cfg,
                                             trust_remote_code=False)
        tok = AutoTokenizer.from_pretrained(self.model_id, revision=self.revision,
                                            trust_remote_code=False)
        return model, tok
