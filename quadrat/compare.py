#!/usr/bin/env python3
"""The comparison across detectors -- one page, every registered adapter, measured or not.

`report.py` renders ONE evaluation. This renders the table the harness exists for, and it is a
different document with different failure modes:

MISSING ROWS ARE THE POINT. A comparison that lists only what finished reads as a complete field
while quietly omitting whatever crashed -- and what crashes is not random, it is the awkward
model. So the roster is built from the ADAPTER REGISTRY, not from the results directory: every
registered detector gets a row, and one that was never measured says so, with the reason, in the
same table as the ones that were. A reader can tell "nobody has run this" from "this scored low".

APERTURES DO NOT MIX. A detector measured through 2000-character chunks and one measured on whole
documents answered different questions, so the aperture is part of the row's identity and is
printed on every line. Rows are grouped by aperture, never silently pooled.

THE SUBJECT IS THE DETECTOR. This page used to end on a table of cells no detector reaches --
the field's collective hole. It was true and it was the wrong emphasis: a reader comes here to
decide about a system, and a list of everyone's shared failures answers a different question. Each
detector's own grid says where IT is blind; that is the finding this page carries.

BINARY DETECTORS ARE NOT RANKED. A signature detector has one operating point; it cannot be moved
to the target FPR the others were placed at. Its row carries its own FPR and is listed apart, so
sorting by recall never puts it in a race it is not running.

ONE BUILD ONLY. Every result records the dataset fingerprint it was computed on. Ids here are
positional, so scores from an earlier build describe different documents; results whose
fingerprint differs from the current corpus are excluded from the tables and listed as stale.

    python3 -m quadrat.compare                       # newest run per (detector, aperture)
    python3 -m quadrat.compare --slice no_pii
    python3 -m quadrat.compare --out reports/comparison

Docs: README.md, section "The comparison page"
"""
from __future__ import annotations

import argparse
import ast
import collections
import datetime
import json
import pathlib

from . import figures as fg
from . import metrics as mt
from . import report as rp

HERE = pathlib.Path(__file__).resolve().parent

from .paths import REPORTS, RESULTS

#: axes compared side by side, in the order a reader needs them: what the injection does, then
#: what it rides in, then what was done to hide it.
#: Empty on purpose. `family` and `action` ARE the grid's rows and columns and are shown by its
#: margins, so a separate section would restate them; the carrier moved into the heat map's
#: subtitle. The remaining axes -- splice position, obfuscation, carrier source -- are caveats
#: about the SET's composition, not claims about a detector: they answer "on which subset was
#: this number taken", and that belongs in the dataset description, not in a comparison of
#: systems.
AXES = []


# --------------------------------------------------------------------------- the roster

def registered() -> dict[str, dict]:
    """Every adapter in `quadrat/detectors/`, by registered name.

    Read from source with `ast`, not by importing. Importing an adapter pulls in torch,
    transformers or a network client, so on a machine without them the roster would silently
    shrink to whatever happened to be installed -- and a detector missing from the comparison
    because of a local dependency is exactly the omission this report exists to prevent."""
    out = {}
    for f in sorted((HERE / "detectors").glob("*.py")):
        if f.name.startswith("_"):
            continue
        try:
            tree = ast.parse(f.read_text(encoding="utf-8"))
        except SyntaxError:
            continue
        doc = (ast.get_docstring(tree) or "").strip().splitlines()
        for node in ast.walk(tree):
            if not isinstance(node, ast.ClassDef):
                continue
            for dec in node.decorator_list:
                if not (isinstance(dec, ast.Call) and getattr(dec.func, "id", "") == "register"):
                    continue
                name = dec.args[0].value if dec.args else node.name
                # `getattr(..., "value", "")`: the scan must survive an adapter that writes
                # something other than a literal here. It used to raise, and the failure
                # surfaced as "the comparative part was not built" -- a whole comparison lost to
                # one unreadable keyword.
                ver = next((getattr(kw.value, "value", "") for kw in dec.keywords
                            if kw.arg == "version"), "")
                binary = any(isinstance(s, ast.Assign)
                             and any(getattr(t, "id", "") == "binary" for t in s.targets)
                             and getattr(s.value, "value", False) is True
                             for s in node.body)
                disp = next((t.value.value for t in node.body
                             if isinstance(t, ast.Assign)
                             and any(getattr(x, "id", "") == "display" for x in t.targets)
                             and isinstance(getattr(t, "value", None), ast.Constant)), "")
                out[name] = {"version": ver, "module": f.name, "binary": binary,
                             "display": disp, "title": doc[0] if doc else ""}
    return out


# --------------------------------------------------------------------------- the results

def title_of(res, reg=None):
    """The system's own name for a heading; the registry name only as a last resort.

    THE LIVE ADAPTER WINS over the name stored in the result. A display name is how a product is
    called, not something the run measured, so renaming an adapter should relabel every page built
    afterwards -- including results recorded before the rename. Reading the result first meant a
    rebuild silently kept the old name on old rows and used the new one on new rows, which is two
    names for one detector in a single comparison. The stored value still answers for a detector
    whose adapter is gone, and results measured before adapters carried `display` have neither, so
    the identifier remains the last resort."""
    return (((reg or {}).get(res.get("detector"), {}) or {}).get("display")
            or res.get("display")
            or res.get("detector", "?"))


def aperture(res: dict) -> str:
    """Short label for what the detector read through. Part of a row's identity.

    A row whose apertures all produced the same profile says so rather than naming one of them:
    for a regex set, chunking is a no-op by construction, and printing "chunks 2000/4" there
    would credit a choice nobody made."""
    same = res.get("_same_apertures")
    if same:
        return "any (" + " = ".join(same) + ")"
    pol = res.get("policy") or "full"
    mark = " ⚑" if res.get("forced_aperture") else ""      # imposed by a flag, not declared
    if pol == "full" or not res.get("window"):
        return "whole document" + mark
    if pol == "chunk":
        return (f"chunks {res['window']}"
                + (f"/{res['overlap']}" if res.get("overlap") else "") + mark)
    return f"{pol} {res['window']}{mark}"


def collect(results_dir, slice_name, dataset=None):
    """Newest result per (detector, aperture) on ONE dataset build.

    Returns (rows, stale, fingerprint). `dataset` pins the build; by default the fingerprint of
    the most recent result wins, which is the current corpus in every normal case."""
    files = sorted(results_dir.glob("*.json"))
    loaded = []
    for f in files:
        try:
            r = json.loads(f.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        if r.get("slice", "all") != slice_name:
            continue
        if r.get("limit"):          # a smoke run over the first N documents, not a measurement
            continue
        # A latency probe carries the same detector, version, aperture and fingerprint as a
        # measurement and differs only in having measured nothing, so it walked in here as a row
        # with no target FPR and took the whole comparison down at the point that spends a budget.
        # Recall is what makes a record a measurement; everything else it shares.
        if "mean_recall" not in r:
            continue
        r["_file"] = f.name
        loaded.append(r)
    if not loaded:
        return [], [], None
    loaded.sort(key=lambda r: r.get("run_at") or "")
    fp = dataset or loaded[-1].get("dataset")

    best, stale = {}, []
    for r in loaded:
        if r.get("dataset") != fp:
            stale.append(r)
            continue
        best[(r.get("detector"), aperture(r))] = r      # sorted by time -> last one wins
    return list(best.values()), stale, fp


# --------------------------------------------------------------------------- shaping

def roster_rows(reg, rows, measured_only=False):
    """Registry x results: one entry per adapter, measured or not.

    MEASURED_ONLY IS FOR A PUBLISHED REPORT. Locally the unmeasured rows are the useful half of
    this table -- an adapter nobody has run is exactly what a reader of their own working copy
    needs pointed out. In a release the same row says something else entirely: it names a third
    party's detector, and all it can report about it is that we have an adapter for it. That is a
    statement about our intentions rather than a measurement, and it ships under a licence and a
    version number as though it were one. A published comparison carries what was measured.
    """
    got = collections.defaultdict(list)
    for r in rows:
        got[r.get("detector")].append(r)
    out = []
    for name, meta in sorted(reg.items()):
        runs = sorted(got.get(name, []), key=lambda r: -r.get("mean_recall", 0))
        if measured_only and not runs:
            continue
        out.append({"name": name, **meta, "runs": runs,
                    "status": "measured" if runs else "adapter present, no measurement"})
    return out


#: Why the matrix is a picture and not a list of numbers: at 92 cells per detector the table of
#: rates is unreadable as a whole, and the finding is a SHAPE -- deep failures sitting beside
#: strong cells, and, across panels, in DIFFERENT PLACES for different systems. If the blind spots
#: coincided they would be a property of the task; that they do not makes them a property of each
#: detector, which is the argument for reporting the cell at all.
HEAT_NOTE = (
    "One grid per detector, same axes in the same order everywhere, one shared scale: more ink is "
    "more recall. Read the panels against each other rather than cell by cell &mdash; the finding "
    "is that the dark and pale regions sit in <em>different places</em> per detector. Blind spots "
    "that coincided would be a property of the corpus; blind spots that move are a property of "
    "the detector, and that is what makes the cell the right unit to report. Every cell carries "
    "its count and interval on hover: at n=80&ndash;240 a single cell is &plusmn;4.5&ndash;8 "
    "points, so the picture is the claim and one cell is not.")

HEAT_NOTE_MD = (
    "One grid per detector, same axes in the same order everywhere; the number is recall in "
    "percent. Read the panels against each other — the finding is that the weak regions sit in "
    "*different places* per detector. Blind spots that coincided would be a property of the "
    "corpus; blind spots that move are a property of the detector. At n=80–240 one cell is "
    "±4.5–8 points, so the pattern is the claim and a single cell is not.")


def cell_axes(rows):
    """(families, actions) in ONE order, shared by every panel.

    The order is a decision, not a detail. Panels can only be compared at a glance if a position
    means the same thing in all of them, so both axes are sorted once, by how the field as a whole
    does on that lever or objective, and every detector is then drawn on that same frame."""
    # Ordered by the POOLED recall of every detector: hits over examples, not a mean of rates.
    # The grid's cells differ in size (80-240), and averaging rates would give a small cell the
    # same weight as one three times larger.
    #
    # GAPS COUNT AS ZEROS -- but ONLY here, and only for the ordering. The grid is incomplete by
    # construction: `guard` lives in mail alone, so its row is shorter than the rest. Scoring a
    # short row over the cells it does have rates it alongside a full one, floating it into the
    # middle of the table and tearing holes through it. Multiplying by the filled share sinks
    # sparse rows and columns to the edges, and the dense part of the grid stays connected.
    #
    # The numbers themselves are untouched: cells and margins carry the real recall over the real
    # n, and a gap stays a dot. Only the order the rows and columns are drawn in changes.
    fam, act = collections.defaultdict(lambda: [0, 0, 0]), collections.defaultdict(lambda: [0, 0, 0])
    for r in rows:
        for cell, v in (r.get("cells") or {}).items():
            f, _, a = cell.partition("/")
            for acc in (fam[f], act[a]):
                acc[0] += v["hits"]
                acc[1] += v["n"]
                acc[2] += 1
    n_acts, n_fams = len(act), len(fam)
    n_rows = max(1, len(rows))

    def score(acc, full):
        """Recall damped by the filled share: gaps enter as zeros."""
        if not acc[1]:
            return 0.0
        return (acc[0] / acc[1]) * (acc[2] / (full * n_rows))

    return (sorted(fam, key=lambda k: -score(fam[k], n_acts)),
            sorted(act, key=lambda k: -score(act[k], n_fams)))


def own_point_rows(ctx, results_dir, root=None):
    """For each BINARY detector whose own rate fits no budget: everyone else, moved to ITS rate.

    A verdict cannot be moved to our operating points, so a detector whose own point sits outside
    them has no column in the table -- and dropping it there would be the wrong lesson, because
    the reason is a property of what its authors shipped, not of how it performs. The comparison
    that IS exact runs the other way: the scored detectors can be placed at its rate, from scores
    already on disk, for nothing. Then every system is at one false-positive rate and the row is a
    like-for-like ranking rather than a caveat.

    Returns [(binary_row, [evaluations of everyone at that rate])]."""
    from . import data as dt
    from . import metrics as mt

    pts = ctx["points"]
    rows = ctx["rows"]
    scored, binary = _sorted_rows(rows, pts[0])
    out = []
    meta = dt.meta_docs(root) if root else dt.meta_docs()
    for b in binary:
        rate = b.get("fpr_pooled") or 0.0
        if any(within_budget(b, t) for t in pts) or not rate:
            continue                      # it already has a column where it belongs
        peers = []
        for r in scored:
            sc = pathlib.Path(results_dir) / (pathlib.Path(r["_file"]).stem + ".scores.jsonl")
            if not sc.exists():
                continue
            try:
                pos, neg = dt.scored(sc, meta=meta)
            except KeyError:
                continue
            ev = mt.evaluate(pos, neg, target_fpr=rate, with_curve=False)
            ev.update(detector=r["detector"], binary=False)
            peers.append(ev)
        if peers:
            out.append((b, sorted(peers + [b], key=lambda e: -e["mean_recall"])))
    return out


def build_figures(ctx, out_dir, theme, stem):
    """Draw the comparison's diagrams into `reports/figures/` and return {key: relative path}.

    Same contract as the single-run report: built once, embedded by both twins, so the picture in
    the Markdown is the file in the HTML rather than a second rendering of the same claim."""
    rows, pts = ctx["rows"], ctx["points"]
    scored, binary = _sorted_rows(rows, pts[0])
    figs = {}
    # LABELS AND KEYS ARE BOTH THE IDENTIFIER on this page. A figure key is an address the
    # renderers agree on; a label used to be the product's own name, and four-word names in a
    # legend, a ranking column and nine heat-map titles is what clipped them. The roster table
    # introduces the mapping once, so the short name is readable everywhere after it.
    # A single detector's own report is a different case: it is about one system, names it in the
    # heading, and has room.
    reg = ctx.get("registry") or {}
    # The identifier, not the product name: the roster table above introduces the mapping once, and
    # a legend of four-word names is a legend that gets clipped.
    nm = lambda r: r["detector"]
    disp = {r["detector"]: nm(r) for r in rows}
    curved = [r for r in scored if r.get("curve")]
    # BINARY DETECTORS GO ON THIS FIGURE TOO, as points with a 95% ellipse. They have no curve --
    # but "no curve" is not "no operating point", and the one view that answers whether a
    # signature detector sits above or below the scored ones was the one view that did not show it.
    if curved or binary:
        figs["curve"] = fg.save(
            fg.curve([(f"{nm(r)} - {aperture(r)}", rp.curve_points(r), None)
                      for r in curved],
                     theme=theme,
                     marks=[rp.binary_mark(dict(r, display=nm(r))) for r in binary],
                     unresolved=(5 / rows[0]["n_negatives"] if rows else None)),
            out_dir, f"{stem}-curve")
    ranked = [(r, pt(r, pts[0]) or r) for r in scored]
    if ranked:
        figs["ranking"] = fg.save(
            # NAME ONLY. The aperture has its own column in the table above, and carrying it
            # into the label pushed every product name past the width the row can draw.
            fg.bars([(nm(r), p["mean_recall"], p.get("mean_ci"),
                      p.get("n_positives", r.get("n_positives", 0)))
                     for r, p in sorted(ranked, key=lambda rp_: rp_[1]["mean_recall"])],
                    theme=theme),
            out_dir, f"{stem}-ranking")
    if binary:
        figs["binary"] = fg.save(
            fg.bars([(f"{nm(r)} - own point", r["mean_recall"], r.get("mean_ci"),
                      r.get("n_positives", 0))
                     for r in sorted(binary, key=lambda r: r["mean_recall"])], theme=theme),
            out_dir, f"{stem}-binary")

    # one grouped figure per marginal axis: same chart type, every detector inside it, so an axis
    # is read ACROSS detectors instead of down the page
    # SCORED DETECTORS ONLY. A binary detector's recall was measured at whatever false-positive
    # rate its own verdict produces, so putting its bar beside detectors placed on a common budget
    # compares numbers taken at different prices. It is not a smaller version of the same
    # comparison, it is a different one -- and the curve above is where it can honestly be made,
    # each system at its own point on one pair of axes.
    names = [nm(r) for r in scored]
    # GROUPS ARE ORDERED BY ONE DETECTOR, the integrally best one -- highest interval AUC, i.e.
    # mean recall over every budget below 1% FPR, the same quantity the aperture was chosen on.
    # Ordering by the best value in each group instead lets the reference jump between detectors
    # from row to row, and then the vertical order encodes nothing: a group is high because
    # somebody, somewhere, was good at it. Against one fixed reference the figure reads as "here
    # is the leader's profile, and here is everyone else against it".
    ref = axis_reference(scored)
    for axis in AXES:
        groups = axis_table(rows, axis)
        if not groups:
            continue
        ordered = axis_order(groups, ref)
        data = [(g, next(iter(v[1] for v in groups[g].values()), 0),
                 {disp.get(d, d): v[0] for d, v in groups[g].items()}) for g in ordered]
        figs[f"axis:{axis}"] = fg.save(fg.bars_grouped(data, names, theme=theme),
                                       out_dir, f"{stem}-{axis}")

    # The sorted profiles ("the shape of a detector") are gone from the reports: fitting a
    # distribution to a detector's per-cell recalls is model material and belongs wherever that
    # model is developed. A report shows the grid and the winner maps -- what reads without one.

    fams, acts = cell_axes(rows)
    if fams and acts:
        for r in rows:
            # A thresholded detector has one grid PER BUDGET, so it gets both in one grid: left
            # half the tight budget, right half the loose one. A binary detector has one point
            # and keeps a whole tile.
            a_cells, b_cells, modes = r.get("cells") or {}, None, ("", "")
            if not r.get("binary"):
                pa, pb = pt(r, pts[0]), pt(r, pts[1] if len(pts) > 1 else pts[0])
                if pa and pa.get("cells"):
                    a_cells = pa["cells"]
                if pb and pb.get("cells") and len(pts) > 1:
                    b_cells = pb["cells"]
                    modes = (f"FPR {pts[0]*100:g}%", f"FPR {pts[1]*100:g}%")
            # Carriers go in the subtitle, not a section of their own: they are the same documents
            # cut another way, and a figure of its own for three numbers would restate the grid a
            # third time. The pooled aggregate sits in the corner of the margins; this is the split.
            m = (r.get("marginals") or {}).get("host_type") or {}
            by_host = " · ".join(f"{h} {m[h]['recall']*100:.0f}%"
                                 for h in ("email", "doc", "web") if h in m)
            sub = (f'{aperture(r)}'
                   + (" - binary, one point" if r.get("binary")
                      else f' - left {modes[0]}, right {modes[1]}' if b_cells else "")
                   + (f' · {by_host}' if by_host else ""))
            figs[f"cells:{r['detector']}"] = fg.save(
                fg.heat(a_cells, fams, acts, nm(r), sub, theme=theme,
                        cells_b=b_cells, modes=modes),
                out_dir, f"{stem}-cells-{r['detector']}")
        # THE FIGURE THE SMALL MULTIPLES COULD NOT MAKE. One grid, coloured by whoever leads the
        # cell: a field with one dominant system comes out one colour, and the patchwork is the
        # claim. Scored detectors only -- a binary one sits at its own rate and leading a cell
        # there means something else.
        # WHO IS ON THE MAP. Every thresholded detector, plus every BINARY one whose own
        # false-positive rate fits inside the budget: it is competing under the same constraint,
        # more strictly in fact, so leaving it out would hand a cell to a scored detector that a
        # signature took at a lower price. A binary row above the budget stays off -- there its
        # recall was bought with false alarms the others were not allowed.
        for b in pts:
            eligible = list(scored) + [r for r in binary if within_budget(r, b)]
            if len(eligible) < 2:
                continue
            names_b = [nm(r) for r in eligible]
            cells_b = {}
            for r in eligible:
                p_ = pt(r, b) if not r.get("binary") else r
                cells_b[nm(r)] = (p_ or r).get("cells") or {}
            extra = [nm(r) for r in binary if within_budget(r, b)]
            sub = f"operating point FPR {b*100:g}% · the number is the leader's recall"
            if extra:
                sub += " · " + ", ".join(extra) + " -- binary, and they fit the budget"
            figs[f"winner:{b}"] = fg.save(
                fg.heat_winner(cells_b, fams, acts, names_b,
                               title="Who leads in each cell", sub=sub, theme=theme),
                out_dir, f"{stem}-winner-{b*100:g}")
        figs["winner"] = figs.get(f"winner:{pts[0]}")
        # ...and one more grid per binary detector that fits no budget, at ITS rate.
        for b, everyone in ctx.get("own_points", []):
            names_o = [disp.get(e["detector"], e["detector"]) for e in everyone]
            figs[f"own:{b['detector']}"] = fg.save(
                fg.heat_winner({disp.get(e["detector"], e["detector"]): (e.get("cells") or {})
                                for e in everyone},
                               fams, acts, names_o, title="Who leads in each cell",
                               sub=f"at {b['detector']}'s own point · "
                                   f"FPR {b['fpr_pooled']*100:.3f}% for everyone",
                               theme=theme),
                out_dir, f"{stem}-own-{b['detector']}")
    return figs


def heat_panels_md(rows, figs):
    """The grids, as pictures only.

    The same numbers used to be repeated as a markdown table under every figure -- 92 cells per
    detector, seven detectors, six hundred numbers nobody reads. A grid IS the picture: the claim
    is a shape, and a table of the same values states it in the form the figure exists to replace.
    Exact per-cell values remain available where they are actually needed: in the result JSON, and
    on hover in the HTML."""
    fams, acts = cell_axes(rows)
    if not fams or not acts:
        return []
    L = ["## Where each detector is blind", ""]
    wins = sorted((float(k.split(":")[1]), v) for k, v in figs.items()
                  if k.startswith("winner:") and v)
    if wins:
        L += ["Which detector leads in each cell, one grid per operating point. One grid rather "
              "than one per system: the claim is about the DIFFERENCE between them, and a panel "
              "with a single dominant detector would come out one flat colour. A pale cell means "
              "a lead too small to read: two cells of 80 need 12-16 points between them before a "
              "difference means anything at 95%, and two of 240 need 7-9. The number is the "
              "leader's recall, so a motley grid of single digits reads as \"everyone is bad "
              "here\". "
              "Binary detectors appear wherever their own false-positive rate fits the budget.", ""]
        for b, path in wins:
            L += [f"![who leads at FPR {b*100:g}%]({path})", ""]
    L += [HEAT_NOTE_MD, ""]
    for r in rows:
        key = f"cells:{r['detector']}"
        if figs.get(key):
            L += [f"![{r['detector']} · {aperture(r)}]({figs[key]})", ""]
    return L


def ensure_derived(rows, results_dir, root=None):
    """Fill in `curve` and the operating points for runs measured before they were recorded.

    THIS IS WHY THE SCORES ARE SAVED. A threshold is a number chosen over a saved score vector,
    so a new operating point, a redrawn curve or a re-cut slice costs a second of arithmetic --
    not another pass over the corpus with a model. Nothing here loads a detector.

    Written back into the result file, so the work happens once. Backfilling is safe precisely
    because the fingerprint already matched: the scores belong to this build, and `data.scored`
    raises on an id that does not, rather than quietly dropping it."""
    from . import data as dt
    from . import metrics as mt

    todo = [r for r in rows if not r.get("binary")
            and (not r.get("curve") or not r.get("points"))]
    if not todo:
        return 0
    meta = dt.meta_docs(root) if root else dt.meta_docs()
    done = 0
    for r in todo:
        sc = pathlib.Path(results_dir) / (pathlib.Path(r["_file"]).stem + ".scores.jsonl")
        if not sc.exists():
            r["_derived_note"] = "scores were not saved"
            continue
        try:
            pos, neg = dt.scored(sc, meta=meta)
        except KeyError as e:
            r["_derived_note"] = str(e)
            continue
        r["curve"] = r.get("curve") or mt.curve(pos, neg)
        r["points"] = r.get("points") or mt.at_points(pos, neg)
        f = pathlib.Path(results_dir) / r["_file"]
        rec = json.loads(f.read_text(encoding="utf-8"))
        rec["curve"], rec["points"] = r["curve"], r["points"]
        f.write_text(json.dumps(rec, ensure_ascii=False, indent=1), encoding="utf-8")
        done += 1
    return done


#: the interval a document-level detector is actually chosen on. Above 1% a filter reading a
#: million pages a day raises ten thousand false alarms a day, which is not an operating point.
INTERVAL_HI = 0.01


def interval_auc(curve, hi=INTERVAL_HI):
    """Mean recall over the budget interval (0, hi], as area under recall vs log10(FPR).

    WHY AN AREA AND NOT A POINT. Two apertures often cross: chunking wins at a tight budget and
    truncation catches up at a loose one, or the reverse. Picking by recall at a single budget
    therefore picks by the budget, and whoever chooses the budget chooses the winner. An area over
    the whole interval a reader would deploy in does not have that degree of freedom.

    log10 because that is the axis the trade-off lives on: 0.01% -> 0.1% is the same practical
    step as 0.1% -> 1%, and a linear area would let the widest decade decide alone. Trapezoid over
    the sampled points, normalised by the log-width actually covered, so a curve sampled more
    densely does not score higher for that reason. Returns None when nothing was sampled inside
    the interval -- a missing comparison, not a zero."""
    import math
    pts = sorted((p for p in (curve or []) if 0 < p["fpr"] <= hi), key=lambda p: p["fpr"])
    if not pts:
        return None
    if len(pts) == 1:
        return pts[0]["recall"]
    area = width = 0.0
    for a, b in zip(pts, pts[1:]):
        w = math.log10(b["fpr"]) - math.log10(a["fpr"])
        area += w * (a["recall"] + b["recall"]) / 2
        width += w
    return area / width if width else pts[-1]["recall"]


def choose_apertures(rows, hi=INTERVAL_HI):
    """One row per detector -- the aperture that wins on (0, hi] -- plus the rows not chosen.

    Every aperture is measured and kept; the choice is only about which one carries the headline.
    The losers are printed with their scores so the pick can be checked rather than trusted."""
    by_det = collections.defaultdict(list)
    for r in rows:
        r["_interval"] = interval_auc(r.get("curve"), hi)
        by_det[r["detector"]].append(r)
    chosen, others = [], []
    for det, rs in by_det.items():
        # THE DECLARED APERTURE IS THE ROW. A detector reads the way its adapter says it reads;
        # a pass through a window a flag imposed answers a question about an integration, not
        # about the detector, so it is listed apart however well it scores.
        declared = [r for r in rs if not r.get("forced_aperture")]
        if declared and len(declared) < len(rs):
            others.extend(r for r in rs if r.get("forced_aperture"))
            rs = declared
        scored = [r for r in rs if r["_interval"] is not None]
        if not scored:
            # A BINARY DETECTOR IS NOT ASKED TO CHOOSE. It has no curve, so there is no interval
            # to win on, and its recall and FPR move together: the aperture that catches more also
            # costs more, and neither dominates. Ranking by recall would silently prefer the more
            # expensive point -- so every aperture keeps its own row, and the reader compares two
            # complete packages instead of a number stripped of its price.
            #
            # The exception is an aperture that changed nothing at all. For a regex set, chunking
            # cannot: a pattern either matches somewhere in the document or it does not, and a max
            # over windows is the same answer. When the profiles come out identical, printing two
            # rows -- or picking one of them -- both suggest a decision that was never available.
            first = rs[0]
            if len(rs) > 1 and all(r["cells"] == first["cells"]
                                   and r.get("fpr") == first.get("fpr") for r in rs[1:]):
                first["_same_apertures"] = [aperture(r) for r in rs]
                chosen.append(first)
            else:
                chosen.extend(rs)
            continue
        best = max(scored, key=lambda r: r["_interval"])
        best["_chosen_over"] = [r for r in rs if r is not best]
        chosen.append(best)
        others.extend(r for r in rs if r is not best)
    return chosen, others


def axis_reference(scored):
    """The detector the marginal axes are ordered by: the integrally best one.

    "Integrally best" is the highest interval AUC -- mean recall over every budget below 1% FPR,
    the same quantity the aperture was chosen on, so the page ranks by one consistent measure
    rather than by whoever happens to lead at the budget being printed."""
    lead = max((r for r in scored if r.get("_interval") is not None),
               key=lambda r: r["_interval"], default=None)
    return lead["detector"] if lead else None


def axis_order(groups, ref):
    """Group labels, best first for `ref`. Ordering by the best value in each group instead lets
    the reference jump between detectors row to row, and then the vertical order encodes nothing:
    a group is high because somebody, somewhere, was good at it."""
    return sorted(groups, key=lambda g: -(groups[g].get(ref, (0, 0))[0] if ref else
                                          max((v[0] for v in groups[g].values()), default=0)))


def axis_table(rows, axis):
    """group -> {detector: (recall, n)} for one marginal axis."""
    groups = {}
    for r in rows:
        for g, v in (r.get("marginals", {}).get(axis) or {}).items():
            groups.setdefault(g, {})[r["detector"]] = (v["recall"], v["n"])
    return groups


# --------------------------------------------------------------------------- rendering

BIN_NOTE = "Thresholded detectors only, all placed on the same budget. A binary detector's recall belongs to the false-positive rate its own verdict produces, so a bar beside these would compare numbers bought at different prices; it is on the budget figure above, at its own point."

BIN_NOTE_MD = 'Thresholded detectors only, all at one budget. A binary detector\'s recall belongs to the false-positive rate its own verdict produces, so a column beside these would compare numbers bought at different prices.'

FLOOR_MD = """`floor` is not a competitor and nobody ships it: five regexes for the most quoted injection phrases, present so that every other number has something to be read against. *Recall 12%* means nothing alone; *recall 12% where five regexes already get 6.5%* means the detector is worth about twice a grep. A detector scoring **below** the floor has not merely come last — it is not detecting this corpus at all. The floor's own recall also reads the corpus back: it is roughly the share of these injections that is quotable boilerplate, and it is small on purpose.

† Binary detector: it emits a verdict, not a score, so it cannot be moved to a budget. Its recall and the false-positive rate beside it are one package, at its own operating point, and the SAME pair is repeated in every budget column -- not because it fits each one, but because it has only the one and cannot be moved to any of them."""

FLOOR_HTML = ("<p class=note><code>floor</code> is not a competitor and nobody ships it: five regexes for the most quoted injection phrases, present so that every other number has something to be read against. &ldquo;Recall 12%&rdquo; means nothing alone; &ldquo;recall 12% where five regexes already get 6.5%&rdquo; means the detector is worth about twice a grep. A detector scoring <b>below</b> the floor has not merely come last &mdash; it is not detecting this corpus at all. The floor's own recall also reads the corpus back: it is roughly the share of these injections that is quotable boilerplate, and it is small on purpose.</p><p class=note>&dagger; Binary detector: it emits a verdict, not a score, so it cannot be moved to a budget. Its recall and the false-positive rate beside it are one package, at its own operating point; the same pair is repeated in every budget column, because it has only the one and cannot be moved to any of them.</p>")


#: How far over a budget a binary detector may sit and still share the column, as a fraction of
#: the budget. Not zero, because a verdict cannot be tuned to a number: demanding an exact match
#: with a rate nobody can hit by construction is false precision, and it would drop a system for
#: three hundredths of a point. Not large either -- the row prints the actual rate, and a reader
#: has to be able to treat the column heading as true. 25% is comfortably outside the budget's own
#: measurement noise (+-8% at 1% FPR on 63000 clean documents) and far short of a doubling.
BUDGET_TOLERANCE = 0.25


def within_budget(res, budget):
    """Can this BINARY detector stand beside the scored ones at `budget`?

    Yes when its own false-positive rate does not exceed the budget. A verdict cannot be moved to
    an operating point, but a detector already firing on FEWER clean documents than the budget
    allows is competing under the same constraint -- more strictly, in fact, and its recall is
    then a LOWER bound at that point: it is not permitted to spend the rest of the budget, and
    could only do better if it were.

    Far above the budget it is out, and that direction is not symmetric: a detector raising many
    more false alarms than the others were allowed is not playing the same game, and putting its
    recall in the same column would flatter it by the amount it overspent. `BUDGET_TOLERANCE`
    is where "slightly over" stops and "not the same game" starts; a detector past it gets its own
    section, where the scored ones are moved to ITS rate instead."""
    return (res.get("fpr_pooled") or 0.0) <= budget * (1 + BUDGET_TOLERANCE)


def _sorted_rows(rows, budget=None):
    """Thresholded rows WORST FIRST, then binary ones -- which are not in the same race.

    ORDERING RULE OF THE REPORT: weakest at the top, strongest at the bottom, everywhere. This
    page is about what detectors miss, and a list that opens with the winner is read as a
    leaderboard -- the eye takes the first row as the answer and the rest as also-rans. Opening
    with the weakest keeps the subject on the range: the reader walks up through the field and
    arrives at the best row already knowing what the floor looks like. It also puts the number a
    reader should be most careful with -- the top one -- last, after the caveats have been read
    rather than before."""
    key = lambda r: (pt(r, budget) or r).get("mean_recall", 0)
    scored = sorted([r for r in rows if not r.get("binary")], key=key)
    binary = sorted([r for r in rows if r.get("binary")], key=key)
    return scored, binary


def pt(res, budget):
    """The evaluation of `res` at a given FPR budget, or None if it was not computed.

    Falls back to the top-level record when that IS the requested budget, so a run recorded
    before the extra points existed still answers for its own primary point."""
    if budget is None:
        return res
    key = f"{budget:g}"
    p = (res.get("points") or {}).get(key)
    if p:
        return p
    return res if res.get("target_fpr") == budget else None


def _cellval(res, budget, field, nd=1, first=False):
    """A cell of the results table.

    A BINARY DETECTOR IS PRINTED IN EVERY COLUMN WHOSE BUDGET IT FITS, and only there. It cannot
    be moved to a budget, but it is either inside one or outside it, and that is per column: a
    detector firing on 0.07% of clean documents is competing under both the 0.1% and the 1% rule,
    so it belongs in both, with a dagger saying the number came from its own point rather than
    from a threshold we set. One firing on 1.11% belongs in the 1% column and NOT in the 0.1% one.

    Printing it once in the leftmost column, as this did before, filed a detector under a budget it
    misses by an order of magnitude -- the dagger explained where the number came from but not that
    the column heading did not apply to it."""
    if res.get("binary"):
        if field not in res or not within_budget(res, budget):
            return "·"
        return f"{rp._pct(res.get(field), nd)}†"
    p = pt(res, budget)
    if p is None:
        return "—"
    v = p.get(field)
    return rp._pct(v, nd) if isinstance(v, (int, float)) else "—"


def _fpr(res, budget, nd=3):
    """THE POOLED RATE, not the worst carrier's.

    This column used to print `max(fpr.values())` -- the highest of the three per-carrier rates --
    beside a recall averaged over the whole pool. Two numbers from two populations in one row: it
    made Picket look like 0.133% (its web carrier) when the pool it was measured on gives 0.086%.
    The threshold is chosen once over every clean document, so the rate that threshold produces is
    one number, and it is this one. Per-
    carrier rates are a separate question and are reported per carrier, where they mean something."""
    p = res if res.get("binary") else pt(res, budget)
    if p is None:
        return "—"
    v = p.get("fpr_pooled")
    if not isinstance(v, (int, float)):
        return "—"
    return rp._pct(v, nd) + ("†" if res.get("binary") else "")


def render_md(ctx, figs=None):
    reg = ctx.get("registry") or {}
    figs = figs or {}
    mdimg = lambda key, alt: ([f"![{alt}]({figs[key]})", ""] if figs.get(key) else [])
    rows, stale, fp = ctx["rows"], ctx["stale"], ctx["dataset"]
    pts = ctx["points"]
    scored, binary = _sorted_rows(rows, pts[0])
    n_pos = rows[0]["n_positives"] if rows else 0
    n_neg = rows[0]["n_negatives"] if rows else 0
    L = ["# Detector comparison", "",
         f"slice `{ctx['slice']}` · {n_pos} unique injections / {n_neg} clean · dataset `{fp}` · "
         f"measured {ctx['measured']}", "",
         "Two operating points, side by side: "
         + " and ".join(f"{b*100:g}% FPR" for b in pts)
         + ". ONE threshold over the whole corpus at each point -- a deployment has one "
         "operating point and does not know a document's carrier in advance; the per-carrier "
         "false-positive rate is therefore reported, not held fixed. Binary detectors are listed "
         "at their own point.",
         "",
         (f"**Built at: {ctx['aperture']}.** " if ctx["aperture"] else "")
         + f"Where a detector was measured through several apertures, the one shown is the one "
           f"that wins over the whole budget interval below {ctx['interval_hi']*100:g}% FPR"
         + (" — the same one for every detector here." if ctx["aperture"]
            else ", and that is not the same aperture for every detector, so it is named per row.")
         # Only promised when it exists. A release ships one aperture per detector, and a sentence
         # pointing at a section that was not rendered sends the reader looking for nothing.
         + (" The rest are in *Other apertures*." if ctx.get("others") else ""), ""]

    # --- roster -------------------------------------------------------------------------
    # THIS TABLE INTRODUCES THE IDENTIFIERS, and every table and figure after it uses only those.
    # A product name is three or four words -- "ProtectAI DeBERTa v3 base v2" -- and repeating it
    # down a ranking column, a curve legend and nine heat-map titles is what made those labels get
    # truncated. Full name once, here, against the short name that carries it everywhere else.
    L += ["## Adapters", "",
          "Every registered adapter, measured or not. **The identifier in the first column is what "
          "every table and figure below uses.**", "",
          "| detector | name | version | status | apertures |", "|---|---|---|---|---|"]
    for e in ctx["roster"]:
        aps = " · ".join(sorted({aperture(r) for r in e["runs"]})) or "—"
        L.append(f"| `{e['name']}` | {e.get('display') or '—'} | {e['version']} | "
                 f"{e['status']} | {aps} |")
    L.append("")

    # --- the table ----------------------------------------------------------------------
    # ONE PAIR OF COLUMNS PER OPERATING POINT: what it catches and what it costs, side by side.
    # A single FPR column for the leftmost budget only made the second point's price invisible --
    # and the price is half of what an operating point means.
    cols = " | ".join(f"recall @{b*100:g}% | FPR @{b*100:g}%" for b in pts)
    L += [f"| detector | aperture | {cols} | coverage | range | worst lever | worst objective |",
          "|---|---|" + "---|---|" * len(pts) + "---|---|---|---|"]
    for r in scored + binary:
        p0 = pt(r, pts[0]) or r
        wf, wa = p0["worst_family"], p0["worst_action"]
        rng = p0["attainable_range"]
        mark = " ⚠binary" if r.get("binary") else ""
        vals = " | ".join(f"{_cellval(r, b, 'mean_recall')} | {_fpr(r, b)}" for b in pts)
        L.append(
            f"| `{r['detector']}`{mark} | {aperture(r)} | {vals} | "
            f"{rp._pct(p0['coverage_50'],0)} ({p0['n_cells']}) | "
            f"{rp._pct(rng[0],0)}–{rp._pct(rng[1],0)} | "
            f"{wf['name']} {rp._pct(wf['recall'])} | {wa['name']} {rp._pct(wa['recall'])} |")
    L.append("")
    L += ["**How to read the table.** `FPR 0.1%` means one false alarm per 1000 clean documents, `FPR 1%` one per 100. On a stream of a million documents a day that is 1000 and 10 000 false alarms. `recall` is the share of injections the detector caught at that threshold; each operating point carries its own pair of \"what it caught / what that cost\".\n\n`coverage` is the share of the 92 grid cells where it catches at least half the injections. `range` runs from the worst cell to the best: the wider it is, the more the detector depends on which kinds of injection it is handed, and the further its average can be moved by changing the mix. `worst lever` and `worst objective` are the weakest technique and the weakest attack goal -- what is left when the adversary picks the class.", ""]

    L += mdimg("ranking", "recall by detector")
    L += mdimg("binary", "binary detectors at their own point")
    L += [FLOOR_MD, ""]

    # --- the curves ----------------------------------------------------------------------
    curved = [r for r in scored if r.get("curve")]
    if curved:
        budgets = [p["target"] for p in curved[0]["curve"]]
        L += ["## Recall against the false-positive budget", "",
              "Every thresholded detector across all budgets at once; the band is the 95% "
              "uncertainty area on both axes. Binary detectors have no curve: they have one point, "
              "drawn as a diamond with an ellipse around it.", ""]
        L += mdimg("curve", "recall against false positive rate")

    # --- apertures not chosen -------------------------------------------------------------
    if ctx["others"]:
        L += ["## Other apertures", "",
              f"Measured and kept, but not carrying the headline row. `interval` is mean recall "
              f"over budgets up to {ctx['interval_hi']*100:g}% FPR — the quantity the choice was "
              "made on, printed so it can be checked.", "",
              # RECALL ONLY, and the header says so. This used to reuse the headline table's
              # `cols` -- recall AND cost per budget, four columns -- while printing two values
              # per row and sizing the separator for a third number of columns. Three widths, one
              # table: it rendered as garbage. An aperture that did not carry the headline is
              # shown for comparison, and the cost is the same at every budget by construction.
              "| detector | aperture | interval | "
              + " | ".join(f"recall @{b*100:g}%" for b in pts) + " | chosen |",
              "|---|---|---|" + "---|" * (len(pts) + 1)]
        for r in sorted(ctx["others"], key=lambda r: (r["detector"], -(r["_interval"] or 0))):
            vals = " | ".join(_cellval(r, b, "mean_recall") for b in pts)
            win = next((c for c in rows if c["detector"] == r["detector"]), None)
            L.append(f"| `{r['detector']}` | {aperture(r)} | "
                     f"{rp._pct(r['_interval']) if r['_interval'] is not None else '—'} | {vals} | "
                     f"{aperture(win) if win else '—'} |")
        L.append("")

    for b, everyone in ctx.get("own_points") or []:
        d = b["detector"]
        L += [f"## At `{d}`'s own point", "",
              f"`{d}` returns a verdict, and its own false-positive rate "
              f"({rp._pct(b['fpr_pooled'], 3)}) falls outside every budget in the table above, so "
              f"it has no column there. The exact comparison runs the other way: every thresholded "
              f"detector is moved to ITS rate from the saved scores. One false-positive rate for "
              f"everyone, no caveats.", "",
              "| detector | recall | worst lever | worst objective |", "|---|---|---|---|"]
        for ev in everyone:
            mark = " ⚠binary" if ev.get("binary") else ""
            L.append(f"| `{ev['detector']}`{mark} | {rp._pct(ev['mean_recall'])} | "
                     f"{ev['worst_family']['name']} {rp._pct(ev['worst_family']['recall'])} | "
                     f"{ev['worst_action']['name']} {rp._pct(ev['worst_action']['recall'])} |")
        L.append("")
        if figs.get(f"own:{d}"):
            L += [f"![who leads at {d}'s point]({figs[f'own:{d}']})", ""]

    # --- the cell matrix, per detector ---------------------------------------------------
    L += heat_panels_md(rows, figs)

    # --- collective blind spots ----------------------------------------------------------
    # --- axis by axis --------------------------------------------------------------------
    names = [r["detector"] for r in scored]
    for axis in AXES:
        groups = axis_table(rows, axis)
        if not groups:
            continue
        title, _note = rp.AXES.get(axis, (axis, ""))
        L += [f"## {title}", "", BIN_NOTE_MD, ""]
        L += mdimg(f"axis:{axis}", title)

    if stale:
        L += ["## Excluded: measured on another build", "",
              "Document ids are positional, so scores from an earlier corpus describe different "
              "documents. These are not comparable and are not shown above.", "",
              "| file | detector | dataset |", "|---|---|---|"]
        for r in stale:
            L.append(f"| `{r['_file']}` | `{r.get('detector')}` | `{r.get('dataset')}` |")
        L.append("")
    return "\n".join(L) + "\n"


def headline_aperture(rows):
    """The aperture the report is built in, or None when the winners disagree.

    Usually every detector wins through the same one, and then the page can say so once in the
    header. When they do not, saying it once would be false, so the header defers to the per-row
    column instead of picking a majority."""
    aps = {aperture(r) for r in rows if not r.get("binary") and not r.get("_same_apertures")}
    return aps.pop() if len(aps) == 1 else None


def peers_at(results_dir, slice_name, target_fpr, exclude, dataset=None, root=None):
    """Every OTHER thresholded detector, re-thresholded to `target_fpr`, for a single report.

    The comparison a per-detector report can honestly carry. A binary detector cannot be moved to
    a budget, but everyone else can be moved to ITS rate -- so its report asks "at the price this
    detector charges in false alarms, what does a scored detector deliver?", which is a question
    with one answer instead of two incomparable ones. THE MOVE IS ALWAYS TOWARD THE FIXED POINT:
    the detector the report is about never has its threshold adjusted to look better against a
    guest, because then the page would be about the comparison rather than about the detector.

    Binary peers are excluded and cannot be included: two detectors at two different self-chosen
    rates are simply two measurements, and no re-thresholding makes them one.

    Returns (peers, skipped). SKIPPED IS NOT AN EMPTY LIST DRESSED UP: a report whose part two
    quietly vanishes reads as "nothing to say", when the truth may be "the only other detector
    here is binary" or "its scores were not saved". Those are different facts about the state of
    this evaluation and both belong on the page.

    Costs a second of arithmetic per peer -- the scores are saved, and a threshold is a number
    chosen over them. Nothing here loads a model."""
    from . import data as dt
    from . import metrics as mt

    rows, _stale, _fp = collect(pathlib.Path(results_dir), slice_name, dataset)
    if not rows:
        return [], []
    ensure_derived(rows, results_dir, root)
    chosen, _others = choose_apertures(rows)
    meta = dt.meta_docs(root) if root else dt.meta_docs()
    out, skipped = [], []
    for r in chosen:
        name = r.get("detector")
        if name == exclude:
            continue
        if r.get("binary"):
            skipped.append((name, "binary: its own point, it does not move to another's"))
            continue
        sc = pathlib.Path(results_dir) / (pathlib.Path(r["_file"]).stem + ".scores.jsonl")
        if not sc.exists():
            skipped.append((name, "scores were not saved -- nothing to re-threshold from"))
            continue
        try:
            pos, neg = dt.scored(sc, meta=meta)
        except KeyError as e:
            skipped.append((name, f"scores do not match this build: {e}"))
            continue
        ev = mt.evaluate(pos, neg, target_fpr=target_fpr, with_curve=True)
        ev.update(detector=r.get("detector"), version=r.get("version"),
                  display=title_of(r, registered()),
                  policy=r.get("policy"), window=r.get("window"), overlap=r.get("overlap"),
                  binary=False, matched_fpr=target_fpr)
        out.append(ev)
    return sorted(out, key=lambda e: e["mean_recall"]), skipped


def build(results_dir=RESULTS, slice_name="all", dataset=None, derive=True,
          points=mt.OPERATING_POINTS, interval_hi=INTERVAL_HI, measured_only=False):
    all_rows, stale, fp = collect(pathlib.Path(results_dir), slice_name, dataset)
    if derive and all_rows:
        n = ensure_derived(all_rows, results_dir)
        if n:
            print(f"curve and operating points backfilled from saved scores: {n} runs")
    rows, others = choose_apertures(all_rows, interval_hi)
    reg = registered()
    ctx = {"rows": rows, "others": others, "all_rows": all_rows, "stale": stale,
            "dataset": fp, "slice": slice_name, "points": list(points),
            "interval_hi": interval_hi, "aperture": headline_aperture(rows),
            "roster": roster_rows(reg, all_rows, measured_only), "registry": reg,
            "own_points": None,   # filled in below; it needs a finished ctx
            # THE NEWEST MEASUREMENT, not the wall clock. A render time made every rebuild
            # produce different bytes for identical inputs -- eleven pages to re-upload for a
            # changed minute -- and it answered a question nobody asks. How fresh the numbers are
            # is a property of the runs.
            "measured": max((r.get("run_at") or "" for r in all_rows), default="")[:16]}
    ctx["own_points"] = own_point_rows(ctx, results_dir)
    return ctx


def write_index(ctx, out_dir, slug_of):
    """`reports/README.md` -- what is in this directory, generated with it.

    The pages are named by run, which is right for a file that has to stay distinguishable across
    rebuilds and wrong for a human opening the directory: ten names like
    `bastion-deberta-v3-xsmall-v1-20260810-131028-chunk2000o4.md` are a wall. The index is the
    translation from the identifier a reader has just met in the comparison to the file that holds
    that detector's profile."""
    reg = ctx.get("registry") or {}
    rows = sorted(ctx["rows"], key=lambda r: -r.get("mean_recall", 0))
    L = ["# Reports", "",
         f"slice `{ctx['slice']}` · dataset `{ctx['dataset']}` · measured {ctx['measured']}", "",
         "**[comparison-all.md](comparison-all.md)** — every detector side by side, the ranking, "
         "the recall-against-budget curve, and who leads in each cell.", "",
         "One page per detector, each carrying its own lever x objective grid, its marginals and "
         "its false positives by carrier:", "",
         "| detector | name | recall @0.1% FPR | page |", "|---|---|---|---|"]
    for r in rows:
        slug = slug_of(r)
        mark = " ⚠binary" if r.get("binary") else ""
        L.append(f"| `{r['detector']}`{mark} | {title_of(r, reg)} | "
                 f"{rp._pct(r.get('mean_recall', 0))} | [{slug}.md]({slug}.md) |")
    others = ctx.get("others") or []
    if others:
        L += ["", "Apertures measured but not carrying the headline row — an aperture imposed by "
                  "a flag answers a question about an integration, not about the detector:", "",
              "| detector | aperture | page |", "|---|---|---|"]
        for r in sorted(others, key=lambda r: r["detector"]):
            slug = slug_of(r)
            L.append(f"| `{r['detector']}` | {aperture(r)} | [{slug}.md]({slug}.md) |")
    L += ["", "Every figure the pages embed is in `figures/`, one SVG per picture: the same file "
              "serves the Markdown here and anything else that includes it.", ""]
    (pathlib.Path(out_dir) / "README.md").write_text("\n".join(L), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser(description="comparison across every registered detector")
    ap.add_argument("--results", default=str(RESULTS))
    ap.add_argument("--slice", default="all")
    ap.add_argument("--dataset", help="pin a fingerprint (default: the newest result's)")
    ap.add_argument("--out", default=None, help="path without extension")
    ap.add_argument("--theme", default="auto", choices=rp.THEMES)
    ap.add_argument("--reports", action="store_true",
                    help="also rebuild every individual report, each with its comparative part")
    ap.add_argument("--no-curves", action="store_true",
                    help="skip backfilling curves and operating points from saved scores")
    ap.add_argument("--interval", type=float, default=INTERVAL_HI,
                    help="upper FPR bound of the interval the aperture is chosen on")
    ap.add_argument("--measured-only", action="store_true",
                    help="leave registered-but-unmeasured adapters out of the roster: what a "
                         "published comparison carries")
    a = ap.parse_args()

    ctx = build(a.results, a.slice, a.dataset, derive=not a.no_curves,
                interval_hi=a.interval, measured_only=a.measured_only)
    if not ctx["rows"]:
        print(f"no results on slice {a.slice}"
              + (f" ({len(ctx['stale'])} stale)" if ctx["stale"] else ""))
        return 1

    out = pathlib.Path(a.out) if a.out else (REPORTS / f"comparison-{a.slice}")
    out.parent.mkdir(parents=True, exist_ok=True)
    # The shared curve is drawn BEFORE the per-detector reports, because each of them embeds it:
    # "across every budget" is one figure with every detector on it, identical on every page, not a
    # private copy per report showing whichever subset that report happened to have in view.
    figs = build_figures(ctx, out.parent, a.theme, out.name)

    if a.reports:
        # Each report's part two compares the others AT ITS OWN RATE, so it is rebuilt here rather
        # than by the runner: only from this side is the full set of saved scores in view.
        # Both the chosen apertures and the ones not chosen: every report on disk should be
        # built by the same code, or the folder holds two generations of the same page.
        reg = ctx.get("registry") or {}
        for r in ctx["rows"] + ctx["others"]:
            # Runs measured before adapters carried `display` have only the registry name in them.
            # Resolving it here rather than in the report keeps one source for the fact -- a page
            # headed by an identifier and a comparison row headed by a product name would be two
            # names for the same detector on two pages built by the same command.
            r["display"] = title_of(r, reg)
            target = r["fpr_pooled"] if r.get("binary") else r.get("target_fpr")
            peers, skipped = peers_at(a.results, a.slice, target, r["detector"],
                                      dataset=ctx["dataset"])
            # BOTH OPERATING POINTS, not just the one this run was launched at. Who leads when a
            # false alarm is expensive and who leads when it is merely unwelcome are two different
            # questions, and the order is not the same in both. A binary detector has one point of
            # its own and gets the single figure.
            pp = []
            if not r.get("binary"):
                for b in ctx["points"]:
                    if rp.at_point(r, b) is None:
                        continue
                    pb, _sk = peers_at(a.results, a.slice, b, r["detector"],
                                       dataset=ctx["dataset"])
                    if pb:
                        pp.append((b, pb))
            slug = pathlib.Path(r["_file"]).stem
            rp.write(r, REPORTS, a.theme, slug=slug,
                     peers=peers, skipped=skipped, peer_points=pp,
                     shared_curve=figs.get("curve"))
        write_index(ctx, REPORTS, lambda r: pathlib.Path(r["_file"]).stem)
        print(f"reports rebuilt: {len(ctx['rows']) + len(ctx['others'])} + index")

    out.with_suffix(".md").write_text(render_md(ctx, figs), encoding="utf-8")

    measured = sum(1 for r in ctx["roster"] if r["runs"])
    print(f"-> {out.with_suffix('.md')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
