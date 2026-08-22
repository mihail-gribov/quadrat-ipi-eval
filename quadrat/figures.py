#!/usr/bin/env python3
"""Figures as standalone SVG files, generated from a result record.

Every diagram in a report is written to `reports/figures/` as its own file and inserted from
there, so one picture serves the HTML report, the Markdown twin and the paper -- one generator,
one set of bytes, no second drawing that can disagree with the first.

WHY FILES AND NOT CSS. The reports used to draw their bars with `<div>`s and CSS. That renders
only inside the HTML page: the Markdown twin had no diagrams at all (42 tables and nothing else),
and nothing could be dropped into a document. An SVG file is the picture, everywhere.

XML, NOT HTML. These files are loaded through `<img>`, so a browser parses them as XML and a
single unquoted attribute makes the whole figure fail to render -- silently, as a broken-image
icon. Every attribute here is quoted for that reason, and `check()` parses each file before it is
written so the failure cannot reach a report.

THEME. A figure behind `<img>` cannot see the host page's variables, so each file carries its own
palette. `auto` writes the light values plus a `prefers-color-scheme: dark` override that the
browser applies to the image on its own; `light`/`dark` bake one palette in, for documents with a
fixed background. The heat ramp is pre-computed steps in classes rather than `color-mix`, so the
dark palette is a class override instead of a second code path.

    from quadrat import figures as fg
    fg.save(fg.bars(rows, "Recall by lever"), out_dir, "lever")

Docs: README.md, section "Figures"
"""
from __future__ import annotations

import html
import math
import pathlib
import xml.etree.ElementTree as ET

#: (ink, muted, line, card, accent, track, ci, heat, heat_lo) per theme.
#: `card` is the page background, not a card colour. A figure is an <img> sitting on the report,
#: so a surface even a shade off the page paints a visible rectangle around every diagram --
#: in dark mode that panel edge was the first thing the eye landed on, before any data.
#:
#: `heat` and `heat_lo` are the two ends of the cell ramp: green where the cell is held, warm
#: where it is open. NOT RED AND GREEN. The pair has to survive deuteranopia -- some 8% of male
#: readers -- and red against green is the one pairing that does not, so the warm end is the rust
#: already used for the diverging figures, whose lightness parts from the green as well as its
#: hue. A reader who cannot separate the hues still reads the ramp as light against dark.
PALETTES = {
    "light": ("#191919", "#78746c", "#e7e4dc", "#fbfaf7", "#8a5a2b", "#eceadf", "#d6cfc0",
              "#2f6f4f", "#a33a2c"),
    "dark": ("#eceae2", "#98958c", "#2e2f27", "#141510", "#d4a373", "#26271f", "#42443a",
             "#4fcf8d", "#ff8f4d"),
}
TOKENS = ("ink", "mut", "line", "card", "accent", "track", "ci", "heat", "heat_lo")

#: One canvas width for every figure, matched to the report's text column: figures of different
#: widths stacked down a page read as a ragged edge before they read as data.
WIDTH = 980

#: Deepest mix of a heat hue under a cell's numeral: the numeral has to survive both ends of the
#: ramp, and 50% is the measured limit at which every step still clears 4.5:1 in both themes -- a
#: deeper ramp needs a second ink colour, and switching ink mid-ramp reads as a category
#: boundary the data does not have. `_contrast_floor` asserts it; the test is in `tests/`.
HEAT_DEPTH = 50
HEAT_STEPS = 21

#: Recall the ramp calls neither held nor open. Not a measured constant -- half the cell caught is
#: the only point on the axis that is nobody's claim, and a midpoint chosen from the data would
#: move every time a detector was added.
HEAT_MID = 0.5

#: Eight lines that stay apart in both themes and survive the common colour-blindness types.
#: Assigned by position, so a detector keeps its colour across every figure on the page.
LINE_COLORS = ("#8a5a2b", "#2f6f4f", "#3b6ea5", "#a33a2c",
               "#6f5a9e", "#7d6b21", "#2f7f7f", "#a3527a")


# --------------------------------------------------------------------------- colour arithmetic

def _s2l(c):
    c /= 255
    return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4


def _l2s(c):
    c = 12.92 * c if c <= 0.0031308 else 1.055 * c ** (1 / 2.4) - 0.055
    return max(0, min(255, round(c * 255)))


def _hex2lin(h):
    h = h.lstrip("#")
    return [_s2l(int(h[i:i + 2], 16)) for i in (0, 2, 4)]


def _lin2oklab(r, g, b):
    l = (0.4122214708 * r + 0.5363325363 * g + 0.0514459929 * b) ** (1 / 3)
    m = (0.2119034982 * r + 0.6806995451 * g + 0.1073969566 * b) ** (1 / 3)
    s = (0.0883024619 * r + 0.2817188376 * g + 0.6299787005 * b) ** (1 / 3)
    return (0.2104542553 * l + 0.7936177850 * m - 0.0040720468 * s,
            1.9779984951 * l - 2.4285922050 * m + 0.4505937099 * s,
            0.0259040371 * l + 0.7827717662 * m - 0.8086757660 * s)


def _oklab2lin(L, a, b):
    l = (L + 0.3963377774 * a + 0.2158037573 * b) ** 3
    m = (L - 0.1055613458 * a - 0.0638541728 * b) ** 3
    s = (L - 0.0894841775 * a - 1.2914855480 * b) ** 3
    return (4.0767416621 * l - 3.3077115913 * m + 0.2309699292 * s,
            -1.2684380046 * l + 2.6097574011 * m - 0.3413193965 * s,
            -0.0041960863 * l - 0.7034186147 * m + 1.7076147010 * s)


def heat_hex(r, theme):
    """The ramp step for rate `r`, as literal hex -- the mix the CSS would make, made here.

    DIVERGING, NOT SEQUENTIAL, and the reason is what a cell means. A cell's recall is not a
    quantity that is simply more or less of something: it says whether that corner of the grid is
    held or open, and those are opposite claims with a hinge between them. On the old single ramp
    the whole difference between a cell nothing catches and a cell caught nine times in ten was
    depth of the same blue, so a grid of holes and a grid of walls had the same colour and
    differed only in how dark it was. Here `HEAT_MID` is the hinge: above it the cell greens,
    below it it warms, and the sign is legible before any numeral is read.

    The midpoint is the page background exactly. A cell sitting at the hinge SHOULD be the
    quietest thing on the grid -- it is the one recall that makes no claim -- and a tint kept
    there for the sake of visibility would be a claim drawn in colour."""
    pal = dict(zip(TOKENS, PALETTES[theme]))
    hue = pal["heat"] if r >= HEAT_MID else pal["heat_lo"]
    # Distance from the hinge, rescaled so each side reaches full depth at its own end of the
    # axis: the two sides are not the same width whenever HEAT_MID is not a half.
    span = (1 - HEAT_MID) if r >= HEAT_MID else HEAT_MID
    d = abs(max(0.0, min(1.0, r)) - HEAT_MID) / span if span else 0.0
    a, b = _lin2oklab(*_hex2lin(hue)), _lin2oklab(*_hex2lin(pal["card"]))
    f = d * HEAT_DEPTH / 100
    return "#%02x%02x%02x" % tuple(
        _l2s(v) for v in _oklab2lin(*[x * f + y * (1 - f) for x, y in zip(a, b)]))


def _relative_luminance(hexc):
    r, g, b = _hex2lin(hexc)
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def contrast(fg_hex, bg_hex):
    """WCAG contrast ratio. Here to be asserted against, not to be drawn with."""
    a, b = _relative_luminance(fg_hex), _relative_luminance(bg_hex)
    lo, hi = sorted((a, b))
    return (hi + 0.05) / (lo + 0.05)


def _contrast_floor(theme):
    """The worst ratio the cell numeral has anywhere on the ramp. Both ends, every step."""
    ink = dict(zip(TOKENS, PALETTES[theme]))["ink"]
    return min(contrast(ink, heat_hex(i / (HEAT_STEPS - 1), theme))
               for i in range(HEAT_STEPS))


# --------------------------------------------------------------------------- svg scaffolding

def _palette_css(theme):
    def block(name):
        vars_ = ";".join(f"--{k}:{v}" for k, v in zip(TOKENS, PALETTES[name]))
        steps = "".join(f".h{i}{{fill:{heat_hex(i / (HEAT_STEPS - 1), name)}}}"
                        for i in range(HEAT_STEPS))
        return f"svg{{{vars_}}}{steps}"

    if theme in PALETTES:
        return block(theme)
    return block("light") + "@media(prefers-color-scheme:dark){" + block("dark") + "}"


BASE_CSS = (
    "svg{font-family:ui-sans-serif,system-ui,-apple-system,'Segoe UI',Roboto,sans-serif}"
    "text{dominant-baseline:middle}"
    ".bg{fill:var(--card)}"
    ".lbl{font-size:11.5px;fill:var(--ink);"
    "font-family:ui-monospace,SFMono-Regular,Menlo,monospace}"
    ".val{font-size:11.5px;fill:var(--ink);font-variant-numeric:tabular-nums;font-weight:600}"
    ".meta{font-size:10px;fill:var(--mut);font-variant-numeric:tabular-nums}"
    ".ttl{font-size:13px;fill:var(--ink);font-weight:600}"
    ".sub{font-size:10.5px;fill:var(--mut)}"
    ".rule{stroke:var(--line);stroke-width:1}"
    ".trk{fill:var(--track)}"
    # The interval rides ON the bar, so it cannot be a tint of the track -- half of it would fall
    # on the fill and vanish. Ink at partial opacity is the one value that stays legible against
    # the track, against the fill, and in both themes.
    ".ci{stroke:var(--ink);stroke-width:1.5;stroke-linecap:butt;opacity:0.55}"
    ".fill{fill:var(--accent)}"
    ".cell{stroke:var(--line);stroke-width:1}"
    ".cellv{font-size:10px;fill:var(--ink);font-variant-numeric:tabular-nums}"
    ".na{font-size:11px;fill:var(--mut)}"
)


def _svg(w, h, theme, body, label):
    return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w:.0f} {h:.0f}" '
            f'width="{w:.0f}" height="{h:.0f}" role="img" '
            f'aria-label="{html.escape(label)}">'
            f'<style>{_palette_css(theme)}{BASE_CSS}</style>'
            f'<rect class="bg" x="0" y="0" width="{w:.0f}" height="{h:.0f}"/>'
            f'{body}</svg>')


def _e(s):
    return html.escape(str(s))


def check(svg):
    """Parse the figure as XML, the way an `<img>` will. Raises on anything a browser would
    silently turn into a broken-image icon."""
    ET.fromstring(svg)
    return svg


# --------------------------------------------------------------------------- bars

ROW_H = 24
BAR_H = 12


def bars(rows, title="", theme="auto", w=WIDTH, baseline=None, baseline_label="",
         highlight=None):
    """Horizontal bars with a confidence interval. `rows` = [(label, recall, (lo, hi), n)].

    `highlight` names the row the page is about. On a ranking of a dozen detectors the reader's
    first question is "which one is mine", and answering it by reading labels is work the figure
    should have done: the named row keeps the full-strength fill while the rest step back, so the
    answer is pre-attentive. Matching is on the part before " - ", because the label carries the
    aperture too and the caller knows only the detector.

    The interval is DRAWN, not only printed: 0 of 80 and 6% of 2240 are the same numeral and not
    the same claim, and only the whisker shows which one is being read. It is a whisker rather
    than a filled band because the band read as a second, paler bar -- two magnitudes on one row
    where there is only one."""
    if not rows:
        return ""
    # Wide enough for a product name: rows are labelled the way the headings are, and
    # "Bastion Prompt Protection" clipped to "Bastion Prompt Protecti" reads as a bug.
    lab_w, val_w, meta_w, gap = 214, 52, 72, 14
    pad_t = 30 if title else 10
    x0 = lab_w + val_w + gap
    tw = w - x0 - meta_w - gap
    h = pad_t + ROW_H * len(rows) + (26 if baseline is not None and baseline_label else 8)
    P = []
    if title:
        P.append(f'<text class="ttl" x="0" y="12">{_e(title)}</text>')
    for i, (label, r, ci, n) in enumerate(rows):
        cy = pad_t + i * ROW_H + ROW_H / 2
        by = cy - BAR_H / 2
        mine = highlight is not None and str(label).split(" - ")[0].strip() == highlight
        dim = '' if (highlight is None or mine) else ' opacity="0.45"'
        short = str(label) if len(str(label)) <= 32 else str(label)[:31] + "\u2026"
        P.append(f'<text class="lbl" x="0" y="{cy:.1f}"'
                 + (' font-weight="600"' if mine else dim)
                 + f'><title>{_e(label)}</title>{_e(short)}</text>')
        P.append(f'<text class="val" x="{lab_w + val_w:.0f}" y="{cy:.1f}" '
                 f'text-anchor="end"{dim}>{r * 100:.1f}%</text>')
        P.append(f'<rect class="trk" x="{x0:.0f}" y="{by:.1f}" width="{tw:.0f}" '
                 f'height="{BAR_H}" rx="{BAR_H / 2:.0f}"/>')
        P.append(f'<rect class="fill" x="{x0:.0f}" y="{by:.1f}" '
                 f'width="{max(2.0, tw * r):.1f}" height="{BAR_H}" rx="{BAR_H / 2:.0f}"{dim}>'
                 f'<title>{_e(label)}: {r * 100:.1f}%'
                 + (f' (CI {ci[0] * 100:.1f}-{ci[1] * 100:.1f})' if ci else '')
                 + f', n={n}</title></rect>')
        if ci:
            a, b = x0 + tw * max(0.0, ci[0]), x0 + tw * min(1.0, ci[1])
            cap = BAR_H * 0.42
            P.append(f'<path class="ci" d="M{a:.1f},{cy - cap:.1f} V{cy + cap:.1f} '
                     f'M{a:.1f},{cy:.1f} H{b:.1f} '
                     f'M{b:.1f},{cy - cap:.1f} V{cy + cap:.1f}" fill="none"/>')
        if baseline is not None:
            bx = x0 + tw * baseline
            P.append(f'<line x1="{bx:.1f}" y1="{by - 3:.1f}" x2="{bx:.1f}" '
                     f'y2="{by + BAR_H + 3:.1f}" stroke="var(--ink)" stroke-width="1.5" '
                     f'opacity="0.4"/>')
        P.append(f'<text class="meta" x="{w:.0f}" y="{cy:.1f}" '
                 f'text-anchor="end">n={n}</text>')
    if baseline is not None and baseline_label:
        y = pad_t + ROW_H * len(rows) + 10
        P.append(f'<line x1="{x0:.0f}" y1="{y - 4:.1f}" x2="{x0:.0f}" y2="{y + 4:.1f}" '
                 f'stroke="var(--ink)" stroke-width="1.5" opacity="0.4"/>')
        P.append(f'<text class="meta" x="{x0 + 8:.0f}" y="{y:.1f}">'
                 f'{_e(baseline_label)} {baseline * 100:.1f}%</text>')
    return check(_svg(w, h, theme, "".join(P), title or "recall by group"))


# --------------------------------------------------------------------------- the cell grid

CELL_W, CELL_H, CELL_GAP = 58, 22, 2


def heat(cells, fams, acts, title, sub="", theme="auto", lab_w=136, w=WIDTH,
         cells_b=None, modes=("", ""), margins=True):
    """The lever x objective grid: one tile per cell, shaded by recall, the rate written in it.

    The axes are passed in, not derived, because every panel of a comparison has to be drawn on
    the SAME axes in the SAME order. Panels exist to be read against each other, and that only
    works if a position means one thing in all of them.

    A THRESHOLDED DETECTOR GETS A SPLIT CELL. It does not have one recall per cell, it has one per
    operating point, and which point you quote is a choice -- so `cells_b` draws the second budget
    as the right half of every tile and the cell shows both at once. Where the halves differ
    sharply, the cell's recall is a fact about the budget rather than about the detector, and that
    is exactly what a single-valued grid hides. A binary detector has one point and keeps a whole
    tile; the shape difference is the reminder that it is not on the same footing."""
    if not fams or not acts:
        return ""
    # THE MARGINS ARE THE MARGINALS. A row's mean recall IS the family's recall and a column's
    # IS the action's: the same hits over the same n. Separate "Recall by lever" and "Recall by
    # objective" sections restated the grid in another shape, so they now live here, against the
    # numbers they summarise.
    n_extra = 1 if margins else 0
    cell_w = max(CELL_W, (w - lab_w - 6) / (len(acts) + n_extra))
    w = lab_w + cell_w * (len(acts) + n_extra) + 6
    top = 40 if title else 22
    h = top + CELL_H * (len(fams) + n_extra) + 8
    P = []
    if title:
        P.append(f'<text class="ttl" x="0" y="11">{_e(title)}</text>')
        if sub:
            P.append(f'<text class="sub" x="{len(title) * 7.4 + 10:.0f}" y="11">'
                     f'{_e(sub)}</text>')
    for j, a in enumerate(acts):
        x = lab_w + cell_w * j + cell_w / 2
        P.append(f'<text class="meta" x="{x:.1f}" y="{top - 11:.0f}" '
                 f'text-anchor="middle">{_e(a[:9])}</text>')
    if margins:
        P.append(f'<text class="meta" x="{lab_w + cell_w * len(acts) + cell_w / 2:.1f}" '
                 f'y="{top - 11:.0f}" text-anchor="middle">lever</text>')
    P.append(f'<line class="rule" x1="0" y1="{top - 3:.0f}" x2="{w:.0f}" '
             f'y2="{top - 3:.0f}"/>')
    for i, f in enumerate(fams):
        y = top + CELL_H * i
        P.append(f'<text class="lbl" x="0" y="{y + CELL_H / 2:.1f}">{_e(f[:18])}</text>')
        for j, a in enumerate(acts):
            x = lab_w + cell_w * j
            v = cells.get(f"{f}/{a}")
            if not v:
                # A dot, not a zero and not an empty tile: this pair is not admitted by the grid.
                # An empty tile would read as "measured, found nothing", a different statement.
                P.append(f'<text class="na" x="{x + cell_w / 2:.1f}" '
                         f'y="{y + CELL_H / 2:.1f}" text-anchor="middle">&#183;</text>')
                continue
            halves = [(v, modes[0])]
            if cells_b is not None:
                halves.append((cells_b.get(f"{f}/{a}"), modes[1]))
            iw_ = (cell_w - CELL_GAP) / len(halves)
            for hi_, (hv, mode) in enumerate(halves):
                hx = x + CELL_GAP / 2 + iw_ * hi_
                if not hv:
                    continue
                step = round(max(0.0, min(1.0, hv["recall"])) * (HEAT_STEPS - 1))
                ci = hv.get("ci") or (0, 0)
                # THE HAIRLINE IS NOT DECORATION. At the hinge the fill IS the page, and without
                # an edge a cell measured at exactly the midpoint looks like the empty space
                # around a cell that the grid does not admit -- two different statements drawn
                # the same way. The border is what keeps "measured, and neither" a tile.
                P.append(f'<rect class="h{step} cell" x="{hx + (0.5 if hi_ else 0):.1f}" '
                         f'y="{y + CELL_GAP / 2:.1f}" width="{iw_ - (1 if len(halves) > 1 else 0):.1f}" '
                         f'height="{CELL_H - CELL_GAP:.0f}" '
                         f'rx="{2 if len(halves) > 1 else 3}">'
                         f'<title>{_e(f)}/{_e(a)}'
                         + (f' at {_e(mode)}' if mode else '')
                         + f': {hv["recall"] * 100:.0f}% ({hv.get("hits", "?")}/{hv["n"]}), '
                         f'CI {ci[0] * 100:.1f}-{ci[1] * 100:.1f}</title></rect>')
                P.append(f'<text class="cellv" x="{hx + iw_ / 2:.1f}" '
                         f'y="{y + CELL_H / 2:.1f}" text-anchor="middle">'
                         f'{hv["recall"] * 100:.0f}</text>')
        if margins:                      # a row's mean IS this lever's recall
            hh = sum(cells[f"{f}/{a}"]["hits"] for a in acts if f"{f}/{a}" in cells)
            nn = sum(cells[f"{f}/{a}"]["n"] for a in acts if f"{f}/{a}" in cells)
            _marg(P, lab_w + cell_w * len(acts), y, cell_w, hh, nn, f)

    if margins:                          # bottom row: a column's mean IS this objective's recall
        y = top + CELL_H * len(fams)
        P.append(f'<text class="lbl" x="0" y="{y + CELL_H / 2:.1f}">objective</text>')
        for j, a in enumerate(acts):
            hh = sum(cells[f"{f}/{a}"]["hits"] for f in fams if f"{f}/{a}" in cells)
            nn = sum(cells[f"{f}/{a}"]["n"] for f in fams if f"{f}/{a}" in cells)
            _marg(P, lab_w + cell_w * j, y, cell_w, hh, nn, a)
        hh = sum(v["hits"] for v in cells.values())
        nn = sum(v["n"] for v in cells.values())
        _marg(P, lab_w + cell_w * len(acts), y, cell_w, hh, nn, "whole set", whole=True)
    return check(_svg(w, h, theme, "".join(P), f"{title}: lever by objective"))


def _marg(P, x, y, cw, hits, n, name, whole=False):
    """A margin cell: the same scale tone, but outlined, so it is not read as an ordinary one.

    A marginal stands on 720-2240 examples against a cell's 80-240 -- an order of magnitude more
    reliable -- and the outline is the reminder that this number is of a different kind."""
    if not n:
        return
    r = hits / n
    step = round(max(0.0, min(1.0, r)) * (HEAT_STEPS - 1))
    P.append(f'<rect class="h{step}" x="{x + CELL_GAP / 2:.1f}" y="{y + CELL_GAP / 2:.1f}" '
             f'width="{cw - CELL_GAP:.0f}" height="{CELL_H - CELL_GAP:.0f}" rx="3" '
             f'stroke="var(--ink)" stroke-opacity="{0.55 if whole else 0.3}" '
             f'stroke-width="1"><title>{_e(name)}: {r * 100:.1f}% ({hits}/{n})</title></rect>')
    P.append(f'<text class="cellv" x="{x + cw / 2:.1f}" y="{y + CELL_H / 2:.1f}" '
             f'text-anchor="middle" font-weight="600">{r * 100:.0f}</text>')


#: A diverging pair: two hues that read as opposite sides of nothing, with a neutral middle.
#: A difference has a sign, and a scale without a midpoint would put "no difference" somewhere
#: along a gradient instead of at a colour of its own.
#:
#: THE SAME PAIR AS THE CELL RAMP, and it used to be a different one -- blue for ahead against
#: the grid's green for held. Two figures on one page, both saying "this is the good side", in
#: two colours: a reader who learned the grid had to learn the difference map separately. These
#: are theme-independent, so they are the mid-lightness members of the pair rather than either
#: palette's endpoints.
DIFF_POS = "#2f6f4f"     # the subject is ahead
DIFF_NEG = "#a33a2c"     # the comparison is ahead
DIFF_MID = "#9a978f"     # within noise


def heat_winner(by_det, fams, acts, names, title="", sub="", theme="auto", lab_w=136, w=WIDTH,
                margin=None):
    """One grid, each cell coloured by WHICH detector leads it.

    The figure the small multiples could not make. "Blind spots sit in different places" asks the
    reader of N separate grids to difference them by eye; here the difference IS the picture. A
    field with one dominant system would come out one colour, and a patchwork is the finding.

    A lead too small to read is drawn pale, and the threshold FOLLOWS THE CELL rather than being
    one flat number. Two independent proportions each carry their own error: at 80 rows a
    difference has to clear roughly 12 points before it means anything at 95%, at 240 rows about 8.
    A single 5-point margin -- what this used to use -- painted a leader in three quarters of the
    grid, and a good half of those were sampling noise dressed as a finding. The number in the cell
    is the leader's recall, so a colourful grid of single digits still reads as "nobody is good
    here"."""
    if not fams or not acts:
        return ""
    cell_w = max(CELL_W, (w - lab_w - 6) / len(acts))
    w = lab_w + cell_w * len(acts) + 6
    top = 40 if title else 22
    # Height follows the SAME row count the legend loop uses; the two drifting apart is
    # how the last legend row gets clipped off the bottom of the file.
    h = top + CELL_H * len(fams) + 22 + 15 * ((len(names) + 2) // 3)
    colour = {n: LINE_COLORS[i % len(LINE_COLORS)] for i, n in enumerate(names)}
    P = []
    if title:
        P.append(f'<text class="ttl" x="0" y="11">{_e(title)}</text>')
        if sub:
            P.append(f'<text class="sub" x="{len(title) * 7.4 + 10:.0f}" y="11">{_e(sub)}</text>')
    for j, a in enumerate(acts):
        P.append(f'<text class="meta" x="{lab_w + cell_w * j + cell_w / 2:.1f}" '
                 f'y="{top - 11:.0f}" text-anchor="middle">{_e(a[:9])}</text>')
    P.append(f'<line class="rule" x1="0" y1="{top - 3:.0f}" x2="{w:.0f}" y2="{top - 3:.0f}"/>')
    for i, f in enumerate(fams):
        y = top + CELL_H * i
        P.append(f'<text class="lbl" x="0" y="{y + CELL_H / 2:.1f}">{_e(f[:18])}</text>')
        for j, a in enumerate(acts):
            x = lab_w + cell_w * j
            got = [(n, (by_det.get(n) or {}).get(f"{f}/{a}")) for n in names]
            size = next((v.get("n") for _, v in got if v), 0)
            got = [(n, v["recall"]) for n, v in got if v]
            # Sized per cell unless the caller insists: 12 points at 80 rows, 8 at 240.
            gap = margin if margin is not None else (0.12 if size and size <= 80 else 0.08)
            if not got:
                P.append(f'<text class="na" x="{x + cell_w / 2:.1f}" y="{y + CELL_H / 2:.1f}" '
                         f'text-anchor="middle">&#183;</text>')
                continue
            got.sort(key=lambda t: -t[1])
            win, best = got[0]
            second = got[1][1] if len(got) > 1 else 0.0
            clear = (best - second) >= gap
            P.append(f'<rect x="{x + CELL_GAP / 2:.1f}" y="{y + CELL_GAP / 2:.1f}" '
                     f'width="{cell_w - CELL_GAP:.0f}" height="{CELL_H - CELL_GAP:.0f}" rx="3" '
                     f'fill="{colour.get(win, DIFF_MID)}" opacity="{0.72 if clear else 0.22}">'
                     f'<title>{_e(f)}/{_e(a)}: {_e(win)} {best * 100:.0f}%'
                     + (f', next {got[1][0]} {second * 100:.0f}%' if len(got) > 1 else '')
                     + ('' if clear else ' (ahead by less than the noise)') + '</title></rect>')
            P.append(f'<text class="cellv" x="{x + cell_w / 2:.1f}" y="{y + CELL_H / 2:.1f}" '
                     f'text-anchor="middle">{best * 100:.0f}</text>')
    # The legend WRAPS. It used to place four per row and never advance y, so a fifth detector
    # was drawn on top of the first -- two names in one place, and the reader counts four systems
    # on a map of five.
    # THREE PER ROW, not four. Detectors are labelled by their product names now, and a quarter of
    # 980 px cut every one of them ("Bastion Prompt Protect", "Regex floor (trivialit") -- a legend
    # that has to be guessed at is worse than one extra row.
    ly = top + CELL_H * len(fams) + 16
    per_row = 3
    for k, n in enumerate(names):
        lx_ = (k % per_row) * (w / per_row)
        row_y = ly + (k // per_row) * 15
        P.append(f'<rect x="{lx_:.0f}" y="{row_y - 5:.0f}" width="10" height="10" rx="2" '
                 f'fill="{colour[n]}" opacity="0.72"/>')
        P.append(f'<text class="meta" x="{lx_ + 15:.0f}" y="{row_y:.0f}">'
                 f'<title>{_e(n)}</title>{_e(n if len(n) <= 30 else n[:29] + chr(8230))}</text>')
    return check(_svg(w, h, theme, "".join(P), "which detector leads each cell"))


def heat_diff(a_cells, b_cells, fams, acts, title="", sub="", theme="auto", lab_w=136, w=WIDTH,
              span=0.4):
    """A - B per cell, on a diverging scale with a neutral middle.

    For the one comparison a per-detector page can make honestly: this detector against another,
    both placed at the same false-positive rate. The sign is the whole content, so the colour has
    a midpoint rather than a gradient -- "no difference" is its own colour, not a shade of one of
    the sides. `span` is the difference that saturates the scale."""
    if not fams or not acts:
        return ""
    cell_w = max(CELL_W, (w - lab_w - 6) / len(acts))
    w = lab_w + cell_w * len(acts) + 6
    top = 40 if title else 22
    h = top + CELL_H * len(fams) + 30
    P = []
    if title:
        P.append(f'<text class="ttl" x="0" y="11">{_e(title)}</text>')
        if sub:
            P.append(f'<text class="sub" x="{len(title) * 7.4 + 10:.0f}" y="11">{_e(sub)}</text>')
    for j, a in enumerate(acts):
        P.append(f'<text class="meta" x="{lab_w + cell_w * j + cell_w / 2:.1f}" '
                 f'y="{top - 11:.0f}" text-anchor="middle">{_e(a[:9])}</text>')
    P.append(f'<line class="rule" x1="0" y1="{top - 3:.0f}" x2="{w:.0f}" y2="{top - 3:.0f}"/>')
    for i, f in enumerate(fams):
        y = top + CELL_H * i
        P.append(f'<text class="lbl" x="0" y="{y + CELL_H / 2:.1f}">{_e(f[:18])}</text>')
        for j, a in enumerate(acts):
            x = lab_w + cell_w * j
            va, vb = (a_cells or {}).get(f"{f}/{a}"), (b_cells or {}).get(f"{f}/{a}")
            if not va or not vb:
                P.append(f'<text class="na" x="{x + cell_w / 2:.1f}" y="{y + CELL_H / 2:.1f}" '
                         f'text-anchor="middle">&#183;</text>')
                continue
            d = va["recall"] - vb["recall"]
            k = min(1.0, abs(d) / span)
            col = DIFF_MID if abs(d) < 0.02 else (DIFF_POS if d > 0 else DIFF_NEG)
            P.append(f'<rect x="{x + CELL_GAP / 2:.1f}" y="{y + CELL_GAP / 2:.1f}" '
                     f'width="{cell_w - CELL_GAP:.0f}" height="{CELL_H - CELL_GAP:.0f}" rx="3" '
                     f'fill="{col}" opacity="{0.15 + 0.6 * k:.2f}">'
                     f'<title>{_e(f)}/{_e(a)}: {va["recall"]*100:.0f}% vs {vb["recall"]*100:.0f}%'
                     f' = {d*100:+.0f} points</title></rect>')
            P.append(f'<text class="cellv" x="{x + cell_w / 2:.1f}" y="{y + CELL_H / 2:.1f}" '
                     f'text-anchor="middle">{d*100:+.0f}</text>')
    ly = top + CELL_H * len(fams) + 16
    for lab, col, off in (("behind", DIFF_NEG, 0), ("level", DIFF_MID, w/4), ("ahead", DIFF_POS, w/2)):
        P.append(f'<rect x="{off:.0f}" y="{ly - 5:.0f}" width="10" height="10" rx="2" '
                 f'fill="{col}" opacity="0.6"/>')
        P.append(f'<text class="meta" x="{off + 15:.0f}" y="{ly:.0f}">{_e(lab)}</text>')
    return check(_svg(w, h, theme, "".join(P), "difference per cell"))


# --------------------------------------------------------------------------- grouped bars

def bars_grouped(groups, names, theme="auto", w=WIDTH, bar_h=9, gap=2):
    """One cluster per group, one bar per detector. `groups` = [(label, n, {name: recall})].

    THE COMPARISON HAS TO BE A CLUSTER, not one chart per detector stacked down the page. Reading
    "who is blind to `inference`" off four separate figures means holding four bar lengths in your
    head and trusting that the axes match; side by side in one cluster it is a glance. The
    detector keeps its colour from the curve above, so identity carries across the page and the
    legend is read once.

    Groups are ordered by their best value, so the bottom of the figure is where the whole field
    is weak -- which is the cell-level finding restated at a resolution that can be read as
    numbers."""
    if not groups or not names:
        return ""
    lab_w, meta_w, pad_r = 156, 64, 14
    row_h = len(names) * (bar_h + gap) + 14
    pad_t = 8
    legend_rows = (len(names) + 2) // 3
    h = pad_t + row_h * len(groups) + 12 + 17 * legend_rows
    x0 = lab_w + 10
    tw = w - x0 - meta_w - pad_r
    P = []
    for i, (label, n, vals) in enumerate(groups):
        y = pad_t + row_h * i
        cy = y + row_h / 2 - 6
        P.append(f'<text class="lbl" x="0" y="{cy:.1f}">{_e(label[:24])}</text>')
        P.append(f'<text class="meta" x="{w:.0f}" y="{cy:.1f}" text-anchor="end">n={n}</text>')
        for k, name in enumerate(names):
            r = vals.get(name)
            if r is None:
                continue
            c = LINE_COLORS[k % len(LINE_COLORS)]
            by = y + k * (bar_h + gap)
            P.append(f'<rect class="trk" x="{x0}" y="{by:.1f}" width="{tw:.0f}" '
                     f'height="{bar_h}" rx="{bar_h / 2:.1f}"/>')
            P.append(f'<rect x="{x0}" y="{by:.1f}" width="{max(2.0, tw * r):.1f}" '
                     f'height="{bar_h}" rx="{bar_h / 2:.1f}" fill="{c}">'
                     f'<title>{_e(name)} on {_e(label)}: {r * 100:.1f}%</title></rect>')
            # with the percent sign: a bare number beside a bar reads as anything -- a count,
            # a rank, an index -- and the first reader asks which
            P.append(f'<text class="meta" x="{x0 + max(2.0, tw * r) + 5:.1f}" '
                     f'y="{by + bar_h / 2:.1f}">{r * 100:.0f}%</text>')
    for k, name in enumerate(names):
        c = LINE_COLORS[k % len(LINE_COLORS)]
        x = x0 + (k % 3) * (tw / 3)
        y = pad_t + row_h * len(groups) + 10 + (k // 3) * 17
        P.append(f'<rect x="{x:.0f}" y="{y - 4:.0f}" width="10" height="8" rx="2" fill="{c}"/>')
        P.append(f'<text class="meta" x="{x + 16:.0f}" y="{y:.0f}">{_e(name[:30])}</text>')
    return check(_svg(w, h, theme, "".join(P), "recall by group, per detector"))


# --------------------------------------------------------------------------- the distribution

def hist(values, marks=(), theme="auto", w=WIDTH, plot_h=170, bins=20,
         xlabel="recall per cell, %"):
    """How one detector's recall is spread ACROSS cells, with weighted means marked on it.

    The figure this harness exists to make. A detector does not have "a recall"; it has this
    spread, and any number quoted from it is a choice of proportions. `marks` = [(label, value,
    colour)] draws those choices on the same axis: two corpora, both honest, both built from these
    very cells, reporting different figures for the same system -- with the distribution behind
    them showing there was never one number to report.

    Bars are counts of cells, so the height is "how many combinations land here" and the width of
    the occupied range is the whole point."""
    if not values:
        return ""
    # THE CANVAS IS COMPUTED FROM THE CONTENT, never assumed. Taking a fixed total height and
    # subtracting padding for the legend silently clipped the last mark off the bottom -- and a
    # clipped figure still renders, so nothing announces it. The plot area is the parameter; the
    # file grows to hold whatever legend the caller asked for.
    pad_l, pad_r, pad_t = 44, 14, 16
    ih, iw = plot_h, w - pad_l - pad_r
    legend_y0 = pad_t + ih + 45
    h = legend_y0 + 16 * max(0, len(marks) - 1) + 12
    counts = [0] * bins
    for v in values:
        counts[min(bins - 1, max(0, int(v * bins)))] += 1
    top = max(counts) or 1
    bw = iw / bins
    P = []
    for i in range(0, 5):
        y = pad_t + ih * (1 - i / 4)
        P.append(f'<line class="rule" x1="{pad_l}" y1="{y:.1f}" x2="{pad_l + iw:.1f}" '
                 f'y2="{y:.1f}"/>')
        P.append(f'<text class="meta" x="{pad_l - 8}" y="{y:.1f}" text-anchor="end">'
                 f'{top * i / 4:.0f}</text>')
    for i, c in enumerate(counts):
        if not c:
            continue
        bh = ih * c / top
        P.append(f'<rect class="fill" x="{pad_l + bw * i + 1:.1f}" '
                 f'y="{pad_t + ih - bh:.1f}" width="{bw - 2:.1f}" height="{bh:.1f}" rx="2">'
                 f'<title>{c} cells at {i * 100 // bins}-{(i + 1) * 100 // bins}%</title></rect>')
    for i in range(0, bins + 1, max(1, bins // 5)):
        x = pad_l + bw * i
        P.append(f'<text class="meta" x="{x:.1f}" y="{pad_t + ih + 13:.0f}" '
                 f'text-anchor="middle">{i * 100 // bins}</text>')
    P.append(f'<text class="meta" x="{pad_l + iw / 2:.0f}" y="{pad_t + ih + 29:.0f}" '
             f'text-anchor="middle">{_e(xlabel)}</text>')
    for k, (label, value, colour) in enumerate(marks):
        c = colour or LINE_COLORS[(k + 2) % len(LINE_COLORS)]
        x = pad_l + iw * min(max(value, 0.0), 1.0)
        P.append(f'<line x1="{x:.1f}" y1="{pad_t - 4}" x2="{x:.1f}" y2="{pad_t + ih:.1f}" '
                 f'stroke="{c}" stroke-width="2" stroke-dasharray="4 3"/>')
        y = legend_y0 + k * 16
        P.append(f'<rect x="{pad_l:.0f}" y="{y - 3:.0f}" width="14" height="3" rx="1.5" '
                 f'fill="{c}"/>')
        P.append(f'<text class="meta" x="{pad_l + 20:.0f}" y="{y:.0f}">'
                 f'{_e(label)} &#8212; {value * 100:.1f}%</text>')
    return check(_svg(w, h, theme, "".join(P), "distribution of recall across cells"))


def profiles(series, theme="auto", w=WIDTH, plot_h=250, marks=()):
    """The fingerprint: per-cell recall sorted descending, one line per detector.

    `series` = [(label, [recall per cell], colour)]. The x axis is a RANK, not a cell -- rank 20
    is a different cell for every detector -- so the line says how competence is distributed, not
    where. Two detectors with the same mean separate here completely: a flat line treats every
    kind of injection alike, a cliff is a system that is excellent at a handful and absent on the
    rest, and the average cannot tell them apart.

    Read three things off it. Where the line starts: the best the detector ever does. How fast it
    falls: how unevenly it looks at types (measured as the half-life of the fitted exponential).
    Where it lands: what an adversary who sends only the worst kind gets to keep."""
    if not series:
        return ""
    # Canvas FROM the content, never a constant with padding subtracted: that arrangement clipped
    # the last legend row here exactly as it clipped the last mark of the histogram, and a clipped
    # figure still renders -- the reader simply counts three detectors on a chart of six.
    legend_rows = (len(series) + 2) // 3
    pad_l, pad_r, pad_t = 46, 14, 14
    iw, ih = w - pad_l - pad_r, plot_h
    legend_y0 = pad_t + ih + 46
    h = legend_y0 + 17 * (legend_rows - 1) + 14
    n = max(len(v) for _, v, _ in series)
    lx = lambda k: pad_l + iw * (k / max(1, n - 1))
    ly = lambda r: pad_t + ih * (1 - min(max(r, 0.0), 1.0))
    P = []
    for frac in (0, .25, .5, .75, 1):
        y = ly(frac)
        P.append(f'<line class="rule" x1="{pad_l}" y1="{y:.1f}" x2="{pad_l + iw:.1f}" '
                 f'y2="{y:.1f}"/>')
        P.append(f'<text class="meta" x="{pad_l - 8}" y="{y:.1f}" text-anchor="end">'
                 f'{frac * 100:g}%</text>')
    for k in range(0, n, max(1, n // 6)):
        P.append(f'<text class="meta" x="{lx(k):.1f}" y="{pad_t + ih + 13:.0f}" '
                 f'text-anchor="middle">{k + 1}</text>')
    P.append(f'<text class="meta" x="{pad_l + iw / 2:.0f}" y="{pad_t + ih + 30:.0f}" '
             f'text-anchor="middle">cells, sorted by descending recall</text>')
    for i, (label, vals, colour) in enumerate(series):
        c = colour or LINE_COLORS[i % len(LINE_COLORS)]
        v = sorted(vals, reverse=True)
        path = " ".join(f"{'M' if j == 0 else 'L'}{lx(j):.1f},{ly(y):.1f}"
                        for j, y in enumerate(v))
        P.append(f'<path d="{path}" fill="none" stroke="{c}" stroke-width="2" '
                 f'stroke-linejoin="round"/>')
        # the level an adversary can force, drawn where the line ends rather than described
        P.append(f'<circle cx="{lx(len(v) - 1):.1f}" cy="{ly(v[-1]):.1f}" r="2.5" fill="{c}"/>')
    for i, (label, _v, colour) in enumerate(series):
        c = colour or LINE_COLORS[i % len(LINE_COLORS)]
        x = pad_l + (i % 3) * (iw / 3)
        y = legend_y0 + (i // 3) * 17
        P.append(f'<rect x="{x:.0f}" y="{y - 2:.0f}" width="16" height="3" rx="1.5" fill="{c}"/>')
        P.append(f'<text class="meta" x="{x + 22:.0f}" y="{y:.0f}">{_e(label[:34])}</text>')
    return check(_svg(w, h, theme, "".join(P), "sorted per-cell recall profile"))


# --------------------------------------------------------------------------- the curve

def curve(series, theme="auto", w=WIDTH, h=360, lo=1e-4, hi=1e-1, marks=(),
          unresolved=None):
    """Recall against the false-positive budget, log-x. `series` = [(label, points, colour)].

    The axis STOPS AT 0.01% on the left, and that is a claim about the corpus, not a style
    choice: 63000 clean documents express rates in steps of 1/63000 = 0.0016%, so a point below
    about 0.01% rests on a handful of false alarms and is not a rate the pool can support. Points
    under the floor are dropped rather than clamped -- clamping would stack them against the left
    edge and draw a measurement that was never made.

    Log-x because the whole question lives in the first decade: the difference between 0.01% and
    0.1% false alarms is the difference between a filter that can run on a firehose and one that
    cannot, and on a linear axis both sit on the origin. Points are plotted at the rate the
    thresholds ACTUALLY produced, which is why a line can step sideways -- ties at the cut move
    the realised rate without moving recall.

    A point may carry `ci` (recall interval) and `fpr_ci`; the recall interval is drawn as a band
    along the line. Without it the lines read as exact, and two of them a point apart look like a
    difference when they are one -- at 16800 injections the interval is about +-0.6 points, which
    is small but is not zero and is not the reader's job to guess.

    `marks` = [(label, fpr, recall, fpr_ci, recall_ci, colour)] puts a BINARY detector on the same
    axes. It has one operating point and no curve, so it is a point with whiskers, never a line:
    interpolating through it would invent a budget it does not offer. Leaving it off the figure
    entirely was worse -- the one view that answers "is the signature detector above or below the
    scored ones at their real operating point" simply did not answer it."""
    series = [s for s in series if s[1]]
    if not series and not marks:
        return ""
    legend_rows = (len(series) + len(marks) + 2) // 3
    pad_l, pad_r, pad_t, pad_b = 50, 14, 14, 46 + 17 * legend_rows
    iw, ih = w - pad_l - pad_r, h - pad_t - pad_b
    span = math.log10(hi) - math.log10(lo)
    lx = lambda f: pad_l + iw * (math.log10(max(f, lo)) - math.log10(lo)) / span
    ly = lambda r: pad_t + ih * (1 - min(max(r, 0.0), 1.0))
    P = []
    # BELOW THE POOL'S RESOLUTION. A rate of 3 false alarms in 63000 clean documents is not a
    # measurement of 0.001% -- the smallest rate the pool can express at all is 1/n, and anything
    # resting on a handful of documents is a number the corpus cannot support. Shaded rather than
    # clipped: the points are real, what is not real is reading them as a rate.
    if unresolved and unresolved > lo:
        xu = lx(unresolved)
        P.append(f'<rect x="{pad_l}" y="{pad_t}" width="{max(0.0, xu - pad_l):.1f}" '
                 f'height="{ih:.1f}" fill="var(--mut)" opacity="0.10"/>')
        P.append(f'<line x1="{xu:.1f}" y1="{pad_t}" x2="{xu:.1f}" y2="{pad_t + ih:.1f}" '
                 f'stroke="var(--mut)" stroke-width="1" stroke-dasharray="3 3"/>')
        P.append(f'<text class="meta" x="{xu - 6:.1f}" y="{pad_t + 10}" text-anchor="end">'
                 f'below the pool\u2019s resolution</text>')
    d = int(math.log10(lo))
    while d <= int(math.log10(hi)):
        x = lx(10 ** d)
        P.append(f'<line class="rule" x1="{x:.1f}" y1="{pad_t}" x2="{x:.1f}" '
                 f'y2="{pad_t + ih:.1f}"/>')
        P.append(f'<text class="meta" x="{x:.1f}" y="{pad_t + ih + 13:.1f}" '
                 f'text-anchor="middle">{10 ** d * 100:g}%</text>')
        d += 1
    for frac in (0, .25, .5, .75, 1):
        y = ly(frac)
        P.append(f'<line class="rule" x1="{pad_l}" y1="{y:.1f}" x2="{pad_l + iw:.1f}" '
                 f'y2="{y:.1f}"/>')
        P.append(f'<text class="meta" x="{pad_l - 8}" y="{y:.1f}" '
                 f'text-anchor="end">{frac * 100:g}%</text>')
    P.append(f'<text class="meta" x="{pad_l + iw / 2:.0f}" y="{pad_t + ih + 30:.1f}" '
             f'text-anchor="middle">false positives (log scale)</text>')
    for i, (label, pts, colour) in enumerate(series):
        pts = [p for p in pts if p.get("fpr", 0) >= lo]
        if not pts:
            continue
        c = colour or LINE_COLORS[i % len(LINE_COLORS)]
        # AN AREA, NOT WHISKERS. The band is one filled polygon: the upper edge runs forward
        # through (FPR_low, recall_high) and the lower edge back through (FPR_high, recall_low),
        # so the area covers the uncertainty on BOTH axes at once. Per-point whiskers scattered
        # the same information across two dozen strokes the reader had to assemble by eye.
        if all(p.get("ci") for p in pts):
            up = [(p.get("fpr_ci", (p["fpr"],))[0] or p["fpr"], p["ci"][1]) for p in pts]
            dn = [(p.get("fpr_ci", (0, p["fpr"]))[1] or p["fpr"], p["ci"][0]) for p in pts]
            band = ([f"{'M' if j == 0 else 'L'}{lx(f):.1f},{ly(r):.1f}"
                     for j, (f, r) in enumerate(up)]
                    + [f"L{lx(f):.1f},{ly(r):.1f}" for f, r in reversed(dn)])
            P.append(f'<path d="{" ".join(band)} Z" fill="{c}" fill-opacity="0.14" '
                     f'stroke="none"/>')
        path = " ".join(f"{'M' if j == 0 else 'L'}{lx(p['fpr']):.1f},{ly(p['recall']):.1f}"
                        for j, p in enumerate(pts))
        P.append(f'<path d="{path}" fill="none" stroke="{c}" stroke-width="2" '
                 f'stroke-linejoin="round" stroke-linecap="round"/>')
        for p in pts:
            # A 2px ring in the surface colour, so overlapping markers stay countable where two
            # detectors cross rather than merging into one blob.
            P.append(f'<circle cx="{lx(p["fpr"]):.1f}" cy="{ly(p["recall"]):.1f}" r="3" '
                     f'fill="{c}" stroke="var(--card)" stroke-width="1.5">'
                     f'<title>{_e(label)}: recall {p["recall"] * 100:.1f}% at FPR '
                     f'{p["fpr"] * 100:.3f}%</title></circle>')
    # A BINARY DETECTOR'S UNCERTAINTY IS AN AREA, not a cross. It has one point and two rates,
    # each with its own interval, and the region the pair could occupy is the ellipse inscribed in
    # them -- a cross of whiskers draws only the two axes of that region and reads as two separate
    # claims. Fitted in DRAWN space, not in rate space: the x axis is logarithmic, so the interval
    # is asymmetric around the point and the ellipse is centred on the midpoint of the transformed
    # bounds, with the marker left at the measured value where it belongs.
    for k, (label, fpr, rec, fpr_ci, rec_ci, colour) in enumerate(marks):
        c = colour or LINE_COLORS[(len(series) + k) % len(LINE_COLORS)]
        x, y = lx(max(fpr, lo)), ly(rec)
        x0 = lx(max(fpr_ci[0], lo)) if fpr_ci and fpr_ci[0] > 0 else x
        x1 = lx(min(fpr_ci[1], hi)) if fpr_ci else x
        y1 = ly(rec_ci[0]) if rec_ci else y
        y0 = ly(rec_ci[1]) if rec_ci else y
        rx, ry = max((x1 - x0) / 2, 3.0), max((y1 - y0) / 2, 3.0)
        P.append(f'<ellipse cx="{(x0 + x1) / 2:.1f}" cy="{(y0 + y1) / 2:.1f}" rx="{rx:.1f}" '
                 f'ry="{ry:.1f}" fill="{c}" opacity="0.22" stroke="{c}" stroke-width="1" '
                 f'stroke-opacity="0.45"/>')
        P.append(f'<rect x="{x - 4:.1f}" y="{y - 4:.1f}" width="8" height="8" rx="1.5" '
                 f'fill="{c}" stroke="var(--card)" stroke-width="1.5" '
                 f'transform="rotate(45 {x:.1f} {y:.1f})">'
                 f'<title>{_e(label)}: recall {rec * 100:.1f}% at FPR {fpr * 100:.3f}% '
                 f'(its own operating point; the ellipse is the 95% region)</title></rect>')

    legend = ([(l, c, False) for l, _p, c in series]
              + [(m[0], m[5], True) for m in marks])
    for i, (label, colour, is_point) in enumerate(legend):
        c = colour or LINE_COLORS[i % len(LINE_COLORS)]
        x = pad_l + (i % 3) * (iw / 3)
        y = pad_t + ih + 46 + (i // 3) * 17
        if is_point:
            P.append(f'<rect x="{x + 4:.0f}" y="{y - 4:.0f}" width="8" height="8" rx="1.5" '
                     f'fill="{c}" transform="rotate(45 {x + 8:.0f} {y:.0f})"/>')
        else:
            P.append(f'<rect x="{x:.0f}" y="{y - 2:.0f}" width="16" height="3" rx="1.5" '
                     f'fill="{c}"/>')
        P.append(f'<text class="meta" x="{x + 22:.0f}" y="{y:.0f}">{_e(label[:34])}</text>')
    return check(_svg(w, h, theme, "".join(P), "recall against false positive rate"))


# --------------------------------------------------------------------------- writing them out

def save(svg, out_dir, name):
    """Write one figure; return its path RELATIVE to the report, which is what both twins embed.
    Returns None for an empty figure, so a caller can skip the block rather than emit a dead
    image tag."""
    if not svg:
        return None
    d = pathlib.Path(out_dir) / "figures"
    d.mkdir(parents=True, exist_ok=True)
    (d / f"{name}.svg").write_text(svg, encoding="utf-8")
    return f"figures/{name}.svg"
