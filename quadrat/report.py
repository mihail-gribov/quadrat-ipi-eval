#!/usr/bin/env python3
"""Render one evaluation as a Markdown report.

MARKDOWN ONLY. There used to be an HTML twin of every page, and keeping the two in step doubled
the work behind every change: a section removed in one and left in the other is two reports that
disagree, and the reader has no way to know which one is current. The figures are files either
way, so the HTML bought nothing the Markdown does not already have.

Two chart types carry the whole report, because nearly every question here has the same shape:

    bars     a sorted list of groups with a recall and an interval. Used for every marginal --
             lever, objective, carrier, placement, source. Sorted best-first, so the tail of the
             list is the answer to "where is it blind".
    heatmap  the lever x objective grid. One per carrier plus the pooled view, so a carrier effect
             reads as the same grid shifting rather than as a single number.

Everything is generated from the result record; no number is written by hand. Reports go outside
the project, so the rendered text is English.
"""
from __future__ import annotations

import collections
import json
import pathlib

from . import figures as fg
from .metrics import OPERATING_POINTS, wilson

THEMES = ("auto", "light", "dark")

#: marginal axis -> (heading, explanatory note). One vocabulary throughout: a document's
#: email/doc/web class is a CARRIER everywhere, never a "stratum" in one place and a "host" in
#: another.
AXES = {
    "family": ("Recall by lever",
               "The construction the injection uses to get obeyed. Sorted best first, so the "
               "bottom of the list is where the detector is blind. Marginals pool ~20x the "
               "examples of a single cell, so these are read as numbers; the grids below are "
               "read as pictures."),
    "action": ("Recall by objective",
               "What happens if the injection succeeds. Usually spreads wider than the lever."),
    "host_type": ("Recall by carrier",
                  "Carriers do not admit the same cells (guard is email-only), so a difference "
                  "here mixes carrier with cell composition. For the carrier effect alone, "
                  "compare the grids below, which hold the cell fixed."),
    "spliced_at": ("Recall by placement",
                   "Where the injection sits in the carrier: appended at the end, as its own "
                   "block between paragraphs, or woven in at a sentence boundary. A detector "
                   "that only reads the tail shows up as a high `end` against low others."),
    "obfuscation": ("Recall by obfuscation",
                    "Character-level distortion that leaves the text readable to a model: "
                    "homoglyphs, zero-width characters, leetspeak, or none. Read these as recall "
                    "ON THE SUBSET each distortion was applied to, not as the price of the "
                    "distortion: obfuscation is skipped where its span would overlap a protected "
                    "region (URL, address, path, code), so the obfuscated rows are a different "
                    "population of injections rather than the same ones distorted. Answering the "
                    "price question needs the pre-obfuscation twin of each row, which v1 does not "
                    "keep."),
    "host_source": ("Recall by carrier source",
                    "Whether the number rests on one corpus."),
}


#: A budget in words. "0.1%" is a number a reader converts in their head and often converts
#: wrong; "one false alarm per 1000 clean documents" is the same fact already converted, and it is
#: what the choice between the two points actually feels like in a queue somebody has to work.
ONE_IN = {0.001: "one false alarm per 1000 clean documents",
          0.01: "one per 100",
          0.0001: "one per 10 000"}



NOTHING_MD = 'Nothing to compare against on this build. The section is kept rather than dropped: one that disappears reads as “no finding”, when what happened is that no second detector has been measured here yet — a fact about this evaluation, not about this detector.'


PEER_NOTE_MD = 'Every detector here was placed at **{fpr}** false positives — the rate `{det}` {why}. The guests were re-thresholded to it from their saved scores; `{det}` itself was not moved. Binary detectors are absent and cannot be added: two systems at two self-chosen rates are two measurements, and no threshold makes them one.'


def _ver_adds(title, ver):
    """Is the version worth printing beside this name?

    `bastion` + `deberta-v3-xsmall-v1` names the checkpoint and is the whole point of pinning it.
    `PIGuard (ACL 2025)` + `acl2025` says the same thing twice, which reads as a version that
    happens to repeat the name rather than as one heading."""
    if not ver:
        return False
    norm = lambda x: str(x).lower().replace("-", "").replace(" ", "").replace(".", "")
    return norm(ver) not in norm(title)


def _pct(x, nd=1):
    return f"{x*100:.{nd}f}%"


def _ci(c):
    return f"{c[0]*100:.1f}–{c[1]*100:.1f}"


#: metric -> what it means to somebody who has never read a detection paper. One sentence, no
#: jargon, and it says what the number is a share OF: "31.7%" alone is unreadable, "31.7% of the
#: 16 800 injections" is not.
MEANING = {
    "recall": "share of the injections it caught",
    "false positives": "how often it flags a clean document — 0.1% is one false alarm per 1000 "
                       "clean documents, 1% is one per 100",
    "coverage": "in how many of the 92 attack types it catches at least half",
    "range": "worst attack type to best — how much the average depends on which types you feed it",
    "weakest lever": "the construction it handles worst; what is left if the attacker picks it",
    "weakest objective": "the goal it handles worst, same reading",
}


def summary(res, points):
    """The headline table as rows of (metric, {budget: value}, meaning).

    BOTH OPERATING POINTS IN ONE TABLE. A thresholded detector has no single recall: it has a
    recall per price in false alarms, and printing the one the run happened to be launched at made
    the page look like a measurement when it was a choice. Side by side the trade is visible --
    Bastion goes 31.7% to 50.7% between the two, which is the whole reason the second point
    exists. A binary detector has one column, its own, and the header says so."""
    def cols(fn):
        return {b: fn(at_point(res, b)) for b in points if at_point(res, b)}

    def rng(p):
        a, b = p["attainable_range"]
        return f"{_pct(a, 0)}–{_pct(b, 0)}"

    rows = [
        ("recall", cols(lambda p: f'{_pct(p["mean_recall"])} · CI {_ci(p["mean_ci"])}'),
         MEANING["recall"] + f' ({res["n_positives"]} unique ones)'),
        ("false positives", cols(lambda p: _pct(p.get("fpr_pooled", 0), 3)),
         MEANING["false positives"] + f' ({res["n_negatives"]} clean documents)'),
        ("coverage", cols(lambda p: f'{_pct(p["coverage_50"])} · '
                                    f'{round(p["coverage_50"] * p["n_cells"])} of {p["n_cells"]}'),
         MEANING["coverage"]),
        ("range over types", cols(rng), MEANING["range"]),
        ("weakest lever", cols(lambda p: f'{p["worst_family"]["name"]} '
                                         f'{_pct(p["worst_family"]["recall"])}'),
         MEANING["weakest lever"]),
        ("weakest objective", cols(lambda p: f'{p["worst_action"]["name"]} '
                                             f'{_pct(p["worst_action"]["recall"])}'),
         MEANING["weakest objective"]),
    ]
    return [r for r in rows if r[1]]


def headline(res):
    """The headline metrics of one result, as an ordered [(key, value, note)].

    One function so the detector and the floor are always described by the same code:
    a comparison column is only honest if both sides were computed identically."""
    rng = res["attainable_range"]
    wf, wa = res["worst_family"], res["worst_action"]
    bf, ba = res.get("best_family"), res.get("best_action")
    out = [
        # Plain rates over every document -- not "mean": they coincide with the mean of the
        # per-cell rates only because the cells are equally filled.
        ("recall", _pct(res["mean_recall"]),
         f'CI {_ci(res["mean_ci"])} · all {res["n_positives"]} unique injections'),
        ("FPR", _pct(res.get("fpr_pooled", 0), 3),
         (f'CI {_ci(res["fpr_pooled_ci"])} · all {res["n_negatives"]} clean'
          if res.get("fpr_pooled_ci") else f'all {res["n_negatives"]} clean')),
        ("coverage r ≥ 0.5", _pct(res["coverage_50"]),
         f'{round(res["coverage_50"] * res["n_cells"])} of {res["n_cells"]} cells'),
        ("attainable range", f'{_pct(rng[0],0)}–{_pct(rng[1],0)}', "over cells"),
    ]
    for label, lo, hi in (("lever", wf, bf), ("objective", wa, ba)):
        if hi:
            out.append((label, f'{_pct(lo["recall"])}–{_pct(hi["recall"])}',
                        f'{lo["name"]} … {hi["name"]}'))
    return out


def curve_points(res):
    """This run's curve with a confidence interval on every point.

    The intervals are NOT stored in the result and do not need to be: a Wilson interval is a
    function of the rate and the count, both of which are already here, so they cost arithmetic
    rather than another pass over the corpus. Without them the line reads as exact, and two lines
    a point apart look like a difference when they are not one."""
    n_pos, n_neg = res.get("n_positives", 0), res.get("n_negatives", 0)
    out = []
    for p in res.get("curve") or []:
        q = dict(p)
        if n_pos:
            q["ci"] = wilson(round(p["recall"] * n_pos), n_pos)
        if n_neg:
            q["fpr_ci"] = wilson(round(p["fpr"] * n_neg), n_neg)
        out.append(q)
    return out


def binary_mark(res, colour=None):
    """A binary detector as one point with whiskers: (label, fpr, recall, fpr_ci, recall_ci, c)."""
    return (f'{res.get("display") or res.get("detector", "binary")} - own point',
            res.get("fpr_pooled", 0.0), res.get("mean_recall", 0.0),
            res.get("fpr_pooled_ci"), res.get("mean_ci"), colour)


def at_point(res, budget):
    """This run's numbers at one false-positive budget, or the run itself if that IS its point.

    A local twin of `compare.pt`, duplicated rather than imported because `compare` imports this
    module and the cycle would only exist to share nine lines."""
    if budget is None:
        return res
    p = (res.get('points') or {}).get(f'{budget:g}')
    if p:
        return p
    return res if res.get('target_fpr') == budget else None


def build_figures(res, out_dir, theme, baseline, slug, peers=(), peer_points=(),
                  shared_curve=None):
    """Draw every diagram of this report into `reports/figures/` and return {key: relative path}.

    Only figures the Markdown actually embeds are drawn. Two used to be produced here and never
    referenced -- a private recall curve per detector, made redundant when every page moved to the
    one shared curve `compare` draws, and a colour scale that lost its only caller when the HTML
    twin went. Together they were 19 files in a release, all of them dead weight."""
    cells, by_host = res["cells"], res.get("cells_by_host", {})
    fams, acts = cell_axes(cells)
    figs = {}

    for h in ("email", "doc", "web"):
        if h in by_host:
            figs[f"cells:{h}"] = fg.save(
                fg.heat(by_host[h], fams, acts, h,
                        f'n={sum(v["n"] for v in by_host[h].values())}', theme=theme),
                out_dir, f"{slug}-cells-{h}")
    figs["cells:all"] = fg.save(
        fg.heat(cells, fams, acts, "all carriers", f'n={res["n_positives"]}', theme=theme),
        out_dir, f"{slug}-cells-all")

    # ---- part two: this detector against the others ------------------------------------
    # The peers arrive ALREADY re-thresholded to this detector's rate (compare.peers_at), so
    # every bar here was bought at the same price in false alarms. The detector the report is
    # about is never moved: it is the fixed point, and the guests come to it.
    if peers:
        # ONE FIGURE PER OPERATING POINT. The two budgets are two different questions -- who leads
        # when a false alarm is expensive, and who leads when it is merely unwelcome -- and the
        # ranking is not the same at both. Drawing only the primary one let the page answer the
        # question the run happened to be launched at.
        me_name = res.get("display") or res.get("detector", "this")
        for budget, plist in (peer_points or [(None, peers)]):
            mine = at_point(res, budget)
            if not mine:
                continue
            rows_ = sorted(
                [(me_name, mine["mean_recall"], mine.get("mean_ci"),
                  mine.get("n_positives", res.get("n_positives", 0)))]
                + [(p.get("display") or p["detector"], p["mean_recall"], p.get("mean_ci"),
                    p.get("n_positives", 0)) for p in plist],
                key=lambda t: -t[1])
            key = "peer:ranking" if budget is None else f"peer:ranking:{budget:g}"
            name = f"{slug}-peers" if budget is None else f"{slug}-peers-{budget:g}"
            figs[key] = fg.save(fg.bars(rows_, theme=theme, highlight=me_name), out_dir, name)
            if budget is not None:
                figs.setdefault("peer:points", []).append((budget, key))

        names = [res.get("detector", "this")] + [p["detector"] for p in peers]
        for axis in ("family", "action"):
            groups = {}
            for name, src in [(res.get("detector", "this"), res)] + [(p["detector"], p)
                                                                     for p in peers]:
                for g, v in (src.get("marginals", {}).get(axis) or {}).items():
                    groups.setdefault(g, {"n": v["n"], "vals": {}})["vals"][name] = v["recall"]
            if not groups:
                continue
            ordered = sorted(groups, key=lambda g: -groups[g]["vals"].get(names[0], 0))
            data = [(g, groups[g]["n"], groups[g]["vals"]) for g in ordered]
            figs[f"peer:{axis}"] = fg.save(fg.bars_grouped(data, names, theme=theme),
                                           out_dir, f"{slug}-peers-{axis}")

    # THE CURVE ACROSS EVERY BUDGET IS NOT BUILT HERE. It is one figure -- every detector on this
    # build, scored ones as lines and binary ones as their own point -- and it is the same figure
    # on every page. Drawing a private copy per report produced a dozen files that differed only in
    # which subset happened to be in view, and a reader comparing two pages could not tell whether
    # a missing line meant "not measured" or "not included here". `compare.build_figures` draws it
    # once and hands the path in.
    if shared_curve:
        figs["peer:curve"] = shared_curve

    # The histogram of per-cell spread ("what this average is an average of") is gone: the grid
    # and the pair "worst marginal / attainable range" already carry that point, and the
    # distribution asked the reader for work the two numbers do for them. The model material
    # (mu, sigma) lives in the experiment, not here.

    return figs


def cell_axes(cells):
    """Both axes ordered by pooled recall, best first, and the SAME order for every carrier panel.

    Alphabetical would scatter the weak cells at random and leave the reader to find the pattern.
    Ordered, the grid reads as a gradient, and a carrier panel that departs from it is visible as
    a departure. Sorting each panel by its own values would destroy exactly that: the panels are
    here to be compared, and comparison needs one frame."""
    by_fam, by_act = collections.defaultdict(list), collections.defaultdict(list)
    for c in cells:
        f, _, a = c.partition("/")
        by_fam[f].append(c)
        by_act[a].append(c)
    mean = lambda ks: sum(cells[k]["recall"] for k in ks) / len(ks)
    return (sorted(by_fam, key=lambda f: -mean(by_fam[f])),
            sorted(by_act, key=lambda a: -mean(by_act[a])))


def render_md(res, baseline=None, figs=None, peers=(), skipped=()):
    figs = figs or {}
    title = res.get("display") or res.get("detector", "?")
    mdimg = lambda key, alt: ([f"![{alt}]({figs[key]})", ""] if figs.get(key) else [])
    det, ver = res.get("detector", "?"), res.get("version", "")
    rng = res["attainable_range"]
    wf, wa = res["worst_family"], res["worst_action"]
    base_head = dict((k, v) for k, v, _n in headline(baseline)) if baseline else {}
    bname = baseline.get("detector", "floor") if baseline else ""
    L = [f"# {title}" + (f" {ver}" if _ver_adds(title, ver) else ""), "",
         f"slice `{res.get('slice','all')}` · {res['n_positives']} unique injections / "
         f"{res['n_negatives']} clean"
         + (f" · dataset `{res['dataset']}`" if res.get("dataset") else ""), ""]
    L += [f"## {title}: the measured profile", ""]
    # A binary detector cannot be placed on a budget, so its column is its own rate; a thresholded
    # one gets a column per operating point. Same table either way, so the two kinds of detector
    # are read the same way.
    if res.get("binary"):
        pts_ = [None]
        heads = [f'its own point (FPR {_pct(res.get("fpr_pooled", 0), 3)})']
    else:
        pts_ = [b for b in OPERATING_POINTS if at_point(res, b)]
        heads = [f"at {b * 100:g}% false positives" for b in pts_]
    rows_ = summary(res, pts_)
    L += ["| metric | " + " | ".join(heads) + " | what it means |",
          "|---|" + "---|" * (len(heads) + 1)]
    for k, vals, why in rows_:
        L.append(f"| {k} | " + " | ".join(str(vals.get(b, "—")) for b in pts_) + f" | {why} |")
    L.append("")

    # ONE threshold, three measured rates. A column repeating the same cut on every row would
    # read as three thresholds -- the per-carrier protocol this harness deliberately does not
    # use -- so the cut is stated once above the table and what varies is what the table shows.
    taus = res.get("threshold") or {}
    tau = next(iter(taus.values()), None) if isinstance(taus, dict) else taus
    L += ["## False positives by carrier", ""]
    if tau is not None and not res.get("binary"):
        L += [f"One threshold over every clean document: **{tau:.4g}**. A deployment does not "
              f"know a document's carrier before reading it, so the rate below is measured per "
              f"carrier, never held fixed there.", ""]
    L += ["| carrier | FPR | 95% CI |", "|---|---|---|"]
    for h in sorted(res["fpr"]):
        L.append(f"| {h} | {_pct(res['fpr'][h],3)} | {_ci(res['fpr_ci'][h])} |")
    L += ["", "## Lever x objective", "",
          "The same grid per carrier, then pooled. A dot is a pair the grid does not admit.", ""]
    for h in ("email", "doc", "web"):
        L += mdimg(f"cells:{h}", f"{h}: lever by objective")
    L += mdimg("cells:all", "all carriers: lever by objective")

    if peers or skipped:
        why = ("its own verdict produces" if res.get("binary")
               else f'was measured at ({res.get("target_fpr", 0) * 100:g}% target)')
        L += ["", f"## {title} against the others, at one false-positive rate", ""]
        if peers:
            L += [PEER_NOTE_MD.format(fpr=_pct(res.get("fpr_pooled", 0), 3),
                                      det=res.get("detector", "this"), why=why), ""]
        if not peers:
            L += [NOTHING_MD, ""]
        for budget, key in (figs.get("peer:points") or []):
            L += [f"### At {budget * 100:g}% false positives — {ONE_IN[budget]}", ""]
            L += mdimg(key, f"recall at {budget * 100:g}% false positives")
        if not figs.get("peer:points"):
            L += mdimg("peer:ranking", "recall at a matched false-positive rate")
        for axis in ("family", "action"):
            if figs.get(f"peer:{axis}"):
                L += [f"### {AXES[axis][0]}, at the same rate", ""]
                L += mdimg(f"peer:{axis}", AXES[axis][0])
        if skipped:
            L += ["Measured on this build but not in the comparison: "
                  + "; ".join(f"`{n}` — {r}" for n, r in skipped) + ".", ""]
        if figs.get("peer:curve"):
            L += ["### Across every budget", "",
                  "Every detector on this build, including the ones that only return a verdict — "
                  "those sit at the rate their own decision produces, drawn as a point with its "
                  "95% region. The same figure appears on every page.", ""]
            L += mdimg("peer:curve", "recall against false positive rate")

    return "\n".join(L) + "\n"


def write(res, out_dir, theme="auto", baseline=None, slug=None, peers=(), skipped=(),
          peer_points=(), shared_curve=None):
    """`slug` comes from the runner so the report sits beside the result it was built from --
    recomputing it here once produced a report whose name silently dropped the overlap."""
    out_dir = pathlib.Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if slug is None:
        tag = res.get("policy", "full")
        if tag != "full":
            tag = f"{tag}{res.get('window')}" + (
                f"o{res['overlap']}" if tag.startswith("chunk") and res.get("overlap") else "")
        slug = f"{res.get('detector','det')}-{res.get('version','')}-{res.get('slice','all')}-{tag}"
    figs = build_figures(res, out_dir, theme, baseline, slug, peers, peer_points, shared_curve)
    (out_dir / f"{slug}.md").write_text(
        render_md(res, baseline, figs, peers, skipped), encoding="utf-8")
    return out_dir / f"{slug}.md"


if __name__ == "__main__":
    import sys
    src = pathlib.Path(sys.argv[1])
    theme = sys.argv[2] if len(sys.argv) > 2 else "auto"
    print(f"-> {write(json.loads(src.read_text()), src.parent / 'reports', theme)}")
