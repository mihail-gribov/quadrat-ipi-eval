#!/usr/bin/env python3
"""Re-score only the documents that changed, splice them into a finished run, recompute.

    python3 -m quadrat.rescore --parent <old-fingerprint> --ids-from data/SCRUB.json
    python3 -m quadrat.rescore --parent <old-fingerprint> --ids pos-015849 --dry-run

WHY THIS IS EXACT, and not an approximation of a re-run. A detector here is deterministic and
reads one document at a time: windows are cut inside a document and never cross into the next, so
a document whose bytes did not change produces the same score it produced before, to the last bit.
Scoring only what changed and keeping the rest is therefore the SAME measurement as a full pass,
not a cheaper estimate of it -- and a four-row edit costs seconds instead of fifteen GPU-hours.

WHAT IS NOT INHERITED: the thresholds. They are re-derived from the whole clean pool, because a
changed negative can move the corpus-wide cut, and a moved cut changes verdicts on documents that
were never touched. Every metric is recomputed from the full spliced score set for that reason;
nothing is carried over from the parent's metrics.

WHAT IS INHERITED, and says so on the record: `seconds` and `n_windows` describe the parent's
full pass. Four documents cannot move either materially, and recomputing them would mean timing a
run that did not happen. `derived_from` names the parent so the chain is auditable.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import time

from .data import DEFAULT_ROOT, fingerprint, load
from .detector import REGISTRY, load_detectors
from .metrics import at_points, evaluate
from .window import WINDOW_512

from .paths import RESULTS as OUT

def spliceable(res, parent_fp, scores_path):
    """Why this run cannot be carried forward, or None if it can."""
    if not scores_path.exists():
        return "scores were not saved"
    # AN UNREGISTERED ADAPTER IS THE WITHHOLDING MECHANISM. A run whose detector is no longer in
    # the registry cannot be re-scored -- there is nothing to score with -- and that is also how a
    # detector is kept out of a release: remove the adapter and its runs stop following the corpus
    # forward. Skipped with a reason rather than raised, so one absent adapter does not stop the
    # other eleven.
    if res.get("detector") not in REGISTRY:
        return "no adapter in the registry"
    if res.get("limit"):
        return "smoke run, not a measurement"
    # THE PARENT MUST BE THE BUILD THIS EDIT WAS MADE TO. Splicing rests on "text unchanged ->
    # score unchanged", which says nothing about a run measured on some other corpus: its
    # untouched documents were different documents. A run from another build is not a stale
    # version of this one, it is an answer to a different question.
    if res.get("dataset") != parent_fp:
        return f"build {res.get('dataset')}, but the edit was made to {parent_fp}"
    return None


def aperture_tag(res):
    """Filesystem-safe form of the aperture, for a run's file name.

    Same facts as `compare.aperture`, without the spaces and slashes a reader wants and a path
    does not. `f` marks an aperture imposed by a flag rather than declared by the adapter."""
    pol = res.get("policy") or "full"
    win = res.get("window")
    parts = [pol if pol != "full" and win else "full"]
    if pol != "full" and win:
        parts.append(str(win))
        if pol == "chunk" and res.get("overlap"):
            parts.append(f"o{res['overlap']}")
    if res.get("forced_aperture"):
        parts.append("f")
    return "".join(parts)


def build_detector(res):
    """The detector instance the parent run measured with, aperture and all.

    The aperture is restored by mutating the instance exactly as `run.py` does, so
    `det.aperture()` answers the same way it answered then. A forced aperture that is not restored
    would silently re-measure the detector's own opening and file the result under the parent's
    settings."""
    name = res["detector"]
    if name not in REGISTRY:
        raise SystemExit(f"no adapter for {name} in the registry -- nothing to re-score with")
    cls, version = REGISTRY[name]
    det = cls()
    det.settings = None
    if res.get("forced_aperture"):
        det.max_chars = res.get("window") or det.max_chars or WINDOW_512
        det.policy = res.get("policy") or "chunk"
        if det.policy == "full":
            det.max_chars = None
    if res.get("overlap") is not None:
        det.overlap = res["overlap"]
    return det, version


def main():
    ap = argparse.ArgumentParser(description="re-score changed documents into finished runs")
    ap.add_argument("--ids", help="comma-separated document ids")
    ap.add_argument("--ids-from", help="JSON with {'rows': [{'id': ...}]} (e.g. SCRUB.json)")
    ap.add_argument("--results", default=str(OUT))
    ap.add_argument("--data", default=str(DEFAULT_ROOT))
    ap.add_argument("--only", help="comma-separated detector names, default: every spliceable run")
    ap.add_argument("--parent", required=True,
                    help="fingerprint of the build the edit was made to; runs on any other "
                         "build are skipped rather than spliced")
    ap.add_argument("--adapters", action="append", default=[])
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    ids = set()
    if a.ids:
        ids |= {s.strip() for s in a.ids.split(",") if s.strip()}
    if a.ids_from:
        ids |= {r["id"] for r in json.loads(pathlib.Path(a.ids_from).read_text())["rows"]}
    if not ids:
        raise SystemExit("nothing to re-score: give --ids or --ids-from")

    load_detectors(a.adapters)
    results = pathlib.Path(a.results)
    fp = fingerprint(a.data)
    print(f"build {fp} · documents to re-score {len(ids)}", flush=True)

    pos, neg = load(a.data)
    by_id = {d.id: d for d in pos + neg}
    missing = ids - set(by_id)
    if missing:
        raise SystemExit(f"not in this build: {sorted(missing)}")
    targets = [by_id[i] for i in sorted(ids)]

    only = {s.strip() for s in a.only.split(",")} if a.only else None
    runs, skipped = [], []
    for f in sorted(results.glob("*.json")):
        try:
            res = json.loads(f.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        if "mean_recall" not in res:            # latency probe and the like, not a measurement
            continue
        sc = results / (f.stem + ".scores.jsonl")
        why = spliceable(res, a.parent, sc)
        if why or (only and res.get("detector") not in only):
            skipped.append((f.name, why or "not in --only"))
            continue
        runs.append((f, res, sc))

    print(f"runs to recompute: {len(runs)}", flush=True)
    for n, why in skipped:
        print(f"  skipped {n}: {why}")
    if a.dry_run:
        for f, res, _ in runs:
            print(f"  {res['detector']:16s} {res.get('policy')}/{res.get('window')} <- {f.name}")
        return 0

    for i, (f, res, sc) in enumerate(runs, 1):
        name = res["detector"]
        print(f"\n[{i}/{len(runs)}] {name} · {res.get('policy')}/{res.get('window')}", flush=True)
        old = {}
        with sc.open(encoding="utf-8") as fh:
            for line in fh:
                r = json.loads(line)
                old[r["id"]] = r["score"]
        absent = set(by_id) - set(old)
        if absent:
            print(f"  ✗ skipped: the scores are missing {len(absent)} documents of this build "
                  f"(e.g. {sorted(absent)[:2]}) -- this run is of another corpus")
            continue

        det, version = build_detector(res)
        det.settings = {k: res.get(k) for k in ("detector", "version", "slice", "policy",
                                                "window", "overlap", "forced_aperture")}
        det.setup()
        t0 = time.time()
        fresh, n_win = det.score_documents(targets)
        det.teardown()
        moved = [(d.id, old[d.id], s) for d, s in zip(targets, fresh) if old[d.id] != s]
        for d, s in zip(targets, fresh):
            old[d.id] = s
        print(f"  re-scored {len(targets)} docs in {time.time() - t0:.1f} s · {n_win} windows · "
              f"score moved on {len(moved)}", flush=True)
        for did, o, s in moved:
            print(f"    {did:18s} {o:.6f} -> {s:.6f}  ({s - o:+.6f})")

        paired_pos = [(d, old[d.id]) for d in pos]
        paired_neg = [(d, old[d.id]) for d in neg]
        binary = bool(res.get("binary"))
        target_fpr = res.get("target_fpr") or 0.001
        out = evaluate(paired_pos, paired_neg, target_fpr=target_fpr, binary=binary)
        out["points"] = {} if binary else at_points(paired_pos, paired_neg)
        # Everything that DESCRIBES the measurement is carried; everything MEASURED is recomputed.
        for k in ("detector", "version", "slice", "policy", "window", "overlap",
                  "forced_aperture", "limit", "pseudonymised", "notes", "display",
                  "seconds", "n_windows"):
            if k in res:
                out[k] = res[k]
        out.update(binary=binary, dataset=fp, run_at=time.strftime("%Y-%m-%d %H:%M:%S"),
                   derived_from=f.name, rescored_ids=sorted(ids),
                   inherited=["seconds", "n_windows"])

        # THE APERTURE IS IN THE NAME, and it has to be. `run.py` keys its slug on the second
        # alone, which is safe there because two apertures of one detector are two multi-hour
        # passes. Here a pass is four documents: both floor apertures finished inside the same
        # second, produced the same slug, and the second write silently replaced the first -- one
        # of twelve runs simply gone, with nothing in the log to say so.
        stamp = time.strftime("%Y%m%d-%H%M%S")
        slug = f"{name}-{version}-{stamp}-{aperture_tag(res)}"
        target = results / f"{slug}.json"
        if target.exists():
            raise SystemExit(f"{target.name} already exists -- refusing to overwrite a run")
        target.write_text(
            json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
        with (results / f"{slug}.scores.jsonl").open("w", encoding="utf-8") as fh:
            for d in pos + neg:
                fh.write(json.dumps({"id": d.id, "score": old[d.id]}) + "\n")
        d_recall = (out["mean_recall"] - res["mean_recall"]) * 100
        d_fpr = (out["fpr_pooled"] - res["fpr_pooled"]) * 100
        print(f"  recall {res['mean_recall']*100:.3f}% -> {out['mean_recall']*100:.3f}% "
              f"({d_recall:+.3f}) · FPR {res['fpr_pooled']*100:.4f}% -> "
              f"{out['fpr_pooled']*100:.4f}% ({d_fpr:+.4f})")
        print(f"  -> {slug}.json", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
