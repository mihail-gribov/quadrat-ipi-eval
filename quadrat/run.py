#!/usr/bin/env python3
"""Run a registered detector over the corpus and print the metrics.

    python3 -m quadrat.run --detector picket
    python3 -m quadrat.run --detector picket --slice no_pii
    python3 -m quadrat.run --list

Raw scores are written next to the result so a re-measure never needs another forward pass: the
threshold, the slice and every metric are recomputed from the saved scores in seconds.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
import time

from .data import DEFAULT_ROOT, SLICES, fingerprint, load
from .detector import REGISTRY, load_detectors
from .metrics import at_points, evaluate, score_floor
from . import report
from .window import OVERLAP_SENTENCES, POLICIES, WINDOW_512, score_windowed

from .paths import REPORTS, RESULTS as OUT


def partial_path(settings):
    """Where a half-finished pass keeps its scores.

    Keyed by the SETTINGS, not by the run's timestamp: a restart is a new run with a new name but
    the same measurement, and it has to find what the previous attempt already paid for."""
    key = json.dumps(settings, sort_keys=True, ensure_ascii=False)
    return OUT / f".partial-{hashlib.sha1(key.encode()).hexdigest()[:12]}.jsonl"


def load_partial(path):
    """({id: score}, windows) from a previous attempt.

    The window count is carried too. Without it a resumed pass recorded only the windows it
    scored itself -- 23600 where the measurement had actually seen 79800 -- and `n_windows` is
    how a reader checks the aperture did what the row claims. A truncated last line is dropped
    rather than guessed."""
    got, wins = {}, 0
    if not path.exists():
        return got, wins
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue                      # killed mid-write; that document is simply redone
            if "_windows" in r:
                wins += r["_windows"]
            else:
                got[r["id"]] = r["score"]
    return got, wins


#: documents per checkpoint. Small enough that a crash loses seconds of work, large enough that
#: the per-batch overhead (a tokenizer warm-up, a connection) stays invisible.
BATCH = int(os.environ.get("QUADRAT_BATCH", "200"))


def score_with_checkpoint(det, docs, path, done, n_win=0):
    """Score what is not already in `done`, writing each batch as it lands.

    IN BATCHES, and that is the whole mechanism: `score_documents` returns a finished list, so
    writing after it returns would record nothing until the pass had already succeeded -- exactly
    the case that needs no checkpoint. Batching is safe because windows never cross a document
    boundary, so a batch is scored identically whether it was passed alone or with the rest.

    WHY THIS EXISTS. A pass over 79800 documents is hours of a card or dollars of somebody's API,
    and a crash at document 79000 used to throw all of it away: `--skip-if-done` matches whole
    passes, and a pass that did not finish wrote nothing at all -- which is precisely the failure
    a long run is most likely to have (a balance running out, a backend redeploy, a laptop lid)."""
    todo = [d for d in docs if d.id not in done]
    if done:
        print(f"  resuming: {len(done)} already scored, {len(todo)} to go", flush=True)
    if todo:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as fh:
            for i in range(0, len(todo), BATCH):
                batch = todo[i:i + BATCH]
                # Where this batch sits in the whole pass. Batching made every adapter's own
                # progress line report its slice ("50/200") instead of the run, which is the one
                # number an operator watches for six hours; these two let it say the truth.
                det.progress_offset = len(done)
                det.progress_total = len(done) + len(todo) - i
                scores, w = det.score_documents(batch)
                n_win += w
                for d, sc in zip(batch, scores):
                    done[d.id] = sc
                    fh.write(json.dumps({"id": d.id, "score": sc}) + "\n")
                fh.write(json.dumps({"_windows": w}) + "\n")
                fh.flush()
    return [done[d.id] for d in docs], n_win


def paired_neg(docs, scores, n_pos):
    return list(zip(docs[n_pos:], scores[n_pos:]))


def check_score_floor(neg, target, name, binary):
    """Refuse to publish an operating point the scores cannot express.

    A tie at the top of the clean distribution puts a floor under the reachable FPR (see
    metrics.score_floor). Asking for a point below that floor yields a threshold that flags the
    whole tie, so the run reports a rate it never achieved -- and nothing about the number looks
    wrong afterwards."""
    if binary:
        return
    tie, n, floor = score_floor(neg)
    # `tie == 1` is not an atom, it is the pool's resolution: with n negatives the smallest
    # expressible rate is 1/n, and on a 40-document smoke that is 7.7% -- blocking every trial run
    # while saying nothing about the scores. An atom means the top VALUE is shared.
    if tie > 1 and floor > target:
        raise SystemExit(
            f"[{name}] target FPR {target*100:g}% is unreachable: {tie} of {n} clean documents "
            f"share the SAME maximal score ({floor*100:.4f}%).\n"
            f"  No threshold separates them, so any point below {floor*100:.4f}% is not a measurement.\n"
            f"  This is score saturation: an fp32 softmax hits its ceiling and a max over many\n"
            f"  windows lands on it. The fix is to keep the logit margin instead of the probability,\n"
            f"  or to use a coarser aperture; see metrics.score_floor.")


def check_score_type(det, scores, name):
    """Refuse to publish a run whose scores contradict how the detector declared itself.

    The declaration decides how the operating point is set, so getting it wrong does not produce a
    slightly-off number, it produces a confident wrong one. A scored detector that actually emits
    verdicts is the dangerous direction: threshold selection sees fewer firings than the FPR budget
    allows, picks tau = 0, and `score >= tau` then flags every document -- which is reported as
    100% recall at 100% FPR and reads like a triumph. That happened here once; hence the check.

    The other direction is only wasteful: a detector declared binary that emits a real score is
    measured at 0.5 instead of at the target FPR, throwing away the resolution it has."""
    uniq = set(scores)
    verdicts = uniq <= {0.0, 1.0}
    if verdicts and not det.binary:
        raise SystemExit(
            f"[{name}] declares a score but returned only {sorted(uniq)} -- a threshold at the target\n"
            f"  FPR degenerates on verdicts (tau=0 flags the whole corpus: 100% recall at 100% FPR).\n"
            f"  If the detector really is binary, set `binary = True` on the adapter class.\n"
            f"  If not, the adapter is binarising the score -- return the continuous value.")
    if not verdicts and det.binary:
        print(f"  ⚠ [{name}] declares binary but returned {len(uniq)} distinct values -- "
              f"measuring at its own point 0.5\n    and not choosing a threshold by FPR; if the score "
              f"is real, drop `binary = True`", flush=True)


def pct(x, nd=1):
    return f"{x * 100:.{nd}f}%"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--detector")
    ap.add_argument("--slice", default="all", choices=sorted(SLICES))
    # The PRIMARY operating point -- the one the console summary and the top level of the record
    # are written at. It is not the only one reported: every pass saves its per-document scores,
    # so `metrics.OPERATING_POINTS` are all evaluated beside it and any other budget can be
    # derived later from the scores alone. Moving an operating point never re-runs a model.
    ap.add_argument("--fpr", type=float, default=0.001, help="target FPR for the single corpus-wide threshold")
    ap.add_argument("--data", default=str(DEFAULT_ROOT))
    ap.add_argument("--limit", type=int, help="smoke test: first N of each class")
    # For a pass whose text leaves the machine. Recorded in the settings, so it is part of the
    # pass's identity: a pseudonymised run and a raw one are different measurements and must not
    # satisfy each other's `--skip-if-done`.
    ap.add_argument("--pseudonymise", action="store_true",
                    help="rewrite carrier e-mail addresses on the way out (injections untouched)")
    # OVERRIDE ONLY. By default the aperture comes from the detector: it declares `max_chars` and
    # splits itself, or declares nothing and gets whole documents. These flags force a different
    # one, which answers a question about an INTEGRATION ("what does a naive caller get?") rather
    # than about the detector, and the forced aperture is recorded in the result so the row cannot
    # be mistaken for the detector's own.
    ap.add_argument("--window", type=int, nargs="?", const=WINDOW_512,
                    help=f"OVERRIDE the detector's aperture: context size in CHARACTERS "
                         f"(bare flag = {WINDOW_512}). Omit to let the detector decide.")
    ap.add_argument("--policy", default=None, choices=POLICIES,
                    help="override how a document longer than the window is handled")
    ap.add_argument("--overlap", type=int, default=None,
                    help=f"override chunk overlap in SENTENCES (default {OVERLAP_SENTENCES})")
    ap.add_argument("--adapters", action="append", default=[],
                    help="directory of detector adapters to load (repeatable)")
    ap.add_argument("--theme", default="auto", choices=report.THEMES,
                    help="force the report's colour scheme (default: follow the reader's)")
    ap.add_argument("--no-report", action="store_true", help="metrics only, skip the HTML")
    ap.add_argument("--skip-if-done", action="store_true",
                    help="exit early if this exact pass was already measured on this dataset")
    ap.add_argument("--list", action="store_true")
    args = ap.parse_args()

    load_detectors(args.adapters)
    if args.list or not args.detector:
        print("registered:")
        for n, (cls, v) in sorted(REGISTRY.items()):
            print(f"  {n:14s} {v:8s} {cls.notes}")
        return
    if args.detector not in REGISTRY:
        raise SystemExit(f"no such detector: {args.detector} (see --list)")

    cls, version = REGISTRY[args.detector]
    det = cls()
    # The adapter can see how the pass was configured -- a judge that ships text off the machine
    # refuses to do it on raw carriers unless that was asked for explicitly.
    det.settings = None

    # Files are stamped with the run's date and time; the settings that define the measurement --
    # policy, window, overlap, slice, dataset fingerprint -- live inside the JSON, where they can
    # be read without parsing a name. Resume therefore looks at content, not at the filename: any
    # prior result whose settings match is a completed pass.
    # The aperture is the DETECTOR's unless a flag overrides it. Applied by mutating the instance
    # rather than by threading parameters through the run: then `det.aperture()` is the single
    # answer to "how was this measured", and the recorded settings cannot drift from what ran.
    forced = bool(args.window or args.policy)
    if forced:
        det.max_chars = args.window or det.max_chars or WINDOW_512
        det.policy = args.policy or "chunk"
        if args.policy in (None, "chunk") and args.overlap is not None:
            det.overlap = args.overlap
        if args.policy == "full":
            det.max_chars = None
    elif args.overlap is not None:
        det.overlap = args.overlap
    policy, window, overlap = det.aperture()

    stamp = time.strftime("%Y%m%d-%H%M%S")   # seconds: two apertures of one detector can start in the same minute
    slug = f"{args.detector}-{version}-{stamp}"
    settings = {"detector": args.detector, "version": version, "slice": args.slice,
                "policy": policy, "window": window, "overlap": overlap,
                # Whether this aperture is the detector's own or one a flag imposed. Without it a
                # forced pass is indistinguishable from a declared one, and an experiment about
                # integrations would sit in the table as the detector's result.
                "forced_aperture": forced,
                # A truncated run is not a smaller version of the measurement, it is a different
                # one -- and it used to be able to displace the real pass, because resume and the
                # comparison both keyed on (detector, aperture) and took the newest. Recorded so
                # `--skip-if-done` will not match across it and the comparison can drop it.
                "limit": args.limit,
                "pseudonymised": args.pseudonymise,
                "dataset": fingerprint(args.data)}
    if args.skip_if_done:
        for prior in OUT.glob(f"{args.detector}-*.json"):
            try:
                got = json.loads(prior.read_text())
            except Exception:
                continue
            if all(got.get(k) == v for k, v in settings.items()):
                print(f"already measured ({prior.name}), skipping")
                return

    t0 = time.time()
    pos, neg = load(args.data, args.slice, args.limit, args.pseudonymise)
    print(f"loaded: {len(pos)} injections, {len(neg)} clean "
          f"(slice {args.slice}, {time.time() - t0:.0f} s)", flush=True)

    # A forced aperture can exceed what a hosted detector will accept. That must stop the run,
    # not be discovered per-request deep inside it: the API adapters return 0.0 on error, so an
    # aperture the service refuses produces a complete, plausible, empty result.
    limit = type(det).max_chars
    if forced and limit and (policy == "full" or window > limit):
        raise SystemExit(
            f"[{args.detector}] accepts at most {limit} chars per document, "
            f"but {'the whole document' if policy == 'full' else window} was asked for.\n"
            f"  The run would not fail: the adapter returns 0.0 on every service refusal, reaches\n"
            f"  the end, and files a plausible result reading 'the detector found nothing'.\n"
            f"  Drop --window (the detector cuts its own) or ask for at most {limit}.")
    det.settings = settings
    det.setup()
    docs = pos + neg
    t1 = time.time()
    # The checkpoint key must include HOW the detector scores, not only what was asked of it.
    # A judge run with JUDGE_LOGPROBS=1 emits probabilities where the same pass without it emitted
    # verdicts; the env var is not part of `settings`, so both would land in one file and mix two
    # scales silently. `binary` is known here because setup() has run.
    part = partial_path({**settings, "binary": det.binary})
    done, seen_win = load_partial(part)
    scores, n_win = score_with_checkpoint(det, docs, part, done, seen_win)
    dt = time.time() - t1
    det.teardown()
    if len(scores) != len(docs):
        raise SystemExit(f"the detector returned {len(scores)} scores for {len(docs)} documents")
    check_score_type(det, scores, args.detector)
    check_score_floor(paired_neg(docs, scores, len(pos)), args.fpr, args.detector, det.binary)
    win = (f"window {window} chars · {policy}"
           + (f" · overlap {overlap} sentences" if policy == "chunk" else "")
           + (" · SET BY FLAG" if forced else " · the detector's own limit")
           if policy != "full" else "whole document (the detector has no limit)")
    print(f"run: {dt:.0f} s  ({len(docs) / dt:.0f} docs/s) · {win} · {n_win} windows", flush=True)

    paired = list(zip(docs, scores))
    res = evaluate(paired[:len(pos)], paired[len(pos):],
                   target_fpr=args.fpr, binary=det.binary)
    # Every budget, computed here because the scores are already in memory. A binary detector
    # gets none: it has one point and no threshold to move.
    res["points"] = ({} if det.binary
                     else at_points(paired[:len(pos)], paired[len(pos):]))
    res.update(**settings, run_at=time.strftime("%Y-%m-%d %H:%M:%S"),
               binary=det.binary, notes=det.notes,
               display=getattr(det, "display", "") or args.detector, seconds=round(dt, 1), n_windows=n_win)

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"{slug}.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    with (OUT / f"{slug}.scores.jsonl").open("w") as fh:
        for d, s in paired:
            fh.write(json.dumps({"id": d.id, "score": s}) + "\n")

    part.unlink(missing_ok=True)      # the finished result supersedes it

    w = res["worst_family"]
    a = res["worst_action"]
    point = ("the detector's own point" if det.binary
             else f"one threshold over all clean documents @ FPR {pct(args.fpr,1)}")
    print(f"\n=== {args.detector} {version} · slice {args.slice} · {point} ===")
    if det.binary:
        print("  ⚠ binary: no threshold is chosen, the FPR is whatever it is; no AUC")
    print(f"  worst family    {pct(w['recall'])}  {w['name']:20s} CI {pct(w['ci'][0])}-{pct(w['ci'][1])}")
    print(f"  worst action    {pct(a['recall'])}  {a['name']:20s} CI {pct(a['ci'][0])}-{pct(a['ci'][1])}")
    print(f"  range           {pct(res['attainable_range'][0],0)} - {pct(res['attainable_range'][1],0)}")
    print(f"  coverage r>=0.5 {pct(res['coverage_50'])}  ({res['n_cells']} cells)")
    print(f"  mean recall     {pct(res['mean_recall'])}  CI {pct(res['mean_ci'][0])}-{pct(res['mean_ci'][1])}")
    print("  FPR by carrier: " + "  ".join(
        f"{h} {pct(res['fpr'][h], 3)}" for h in sorted(res["fpr"])))
    print("\n  marginal by carrier:")
    for h, v in sorted(res["marginals"]["host_type"].items(), key=lambda kv: kv[1]["recall"]):
        print(f"    {h:6s} {pct(v['recall']):>7s}  n={v['n']:5d}  CI {pct(v['ci'][0])}-{pct(v['ci'][1])}")
    print("\n  worst cells:")
    for c in res["worst_cells"][:8]:
        print(f"    {c['cell']:36s} {pct(c['recall']):>7s}  n={c['n']}")
    print(f"\n-> {OUT / (slug + '.json')}")

    # Part 2 of the report: the other detectors, re-thresholded to THIS one's rate. Guarded --
    # a comparison that cannot be built must not take the measurement down with it.
    peers, skipped = (), ()
    if not args.no_report:
        try:
            from .compare import peers_at
            peers, skipped = peers_at(OUT, args.slice, res.get("fpr_pooled") if det.binary
                                      else args.fpr, args.detector,
                                      dataset=settings["dataset"], root=args.data)
        except Exception as e:
            print(f"  the comparative part was not built: {e!r}", flush=True)

    if not args.no_report:
        # Compare against the floor whenever one was measured on the SAME dataset and aperture --
        # a floor from another corpus or window would make the delta meaningless.
        base = None
        # The floor is comparable only through the same aperture and on the same dataset; the
        # newest such run wins if several exist.
        if args.detector != "floor":
            want = {k: settings[k] for k in ("slice", "policy", "window", "overlap", "dataset")}
            cands = []
            for f in OUT.glob("floor-*.json"):
                try:
                    c = json.loads(f.read_text())
                except Exception:
                    continue
                if all(c.get(k) == v for k, v in want.items()):
                    cands.append((c.get("run_at", ""), c))
            if cands:
                base = max(cands)[1]
        print(f"-> {report.write(res, REPORTS, args.theme, base, slug=slug, peers=peers, skipped=skipped)}")


if __name__ == "__main__":
    main()
