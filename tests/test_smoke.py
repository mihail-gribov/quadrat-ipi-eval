#!/usr/bin/env python3
"""Does the harness work at all, on a machine that is not the author's?

Every test here runs the real entry points as a user would -- `python -m quadrat.run`,
`python -m quadrat.compare` -- against a tiny corpus in a temporary directory. Nothing is mocked
and no model is downloaded, so the suite finishes in seconds and still fails if installation,
paths, the registry, the metrics or the report generator are broken.

    uv run pytest
"""
from __future__ import annotations

import json
import os
import pathlib
import subprocess
import sys

import pytest

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

import corpus  # noqa: E402


def run(args, data, out, check=True):
    env = {**os.environ, "QUADRAT_DATA": str(data), "QUADRAT_OUT": str(out),
           "PYTHONPATH": str(ROOT)}
    r = subprocess.run([sys.executable, "-m", *args], cwd=ROOT, env=env,
                       capture_output=True, text=True)
    if check and r.returncode:
        raise AssertionError(f"{' '.join(args)} failed:\n{r.stdout}\n{r.stderr}")
    return r


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    """One corpus, one pass of each detector, one comparison -- shared by every assertion."""
    base = tmp_path_factory.mktemp("quadrat")
    data, out = base / "data", base / "out"
    n_pos, n_neg = corpus.write(data)
    run(["quadrat.run", "--detector", "floor"], data, out)
    run(["quadrat.run", "--detector", "graded", "--adapters", str(HERE / "adapters")], data, out)
    run(["quadrat.compare", "--reports"], data, out)
    return {"data": data, "out": out, "n_pos": n_pos, "n_neg": n_neg}


def results(out, detector):
    files = sorted((out / "results").glob(f"{detector}-*.json"))
    assert files, f"no result written for {detector}"
    return json.loads(files[-1].read_text(encoding="utf-8"))


# --------------------------------------------------------------------------- the harness runs

def test_registry_needs_no_model(built):
    """The roster is read from source, so it must list adapters whose dependencies are absent."""
    from quadrat.compare import registered
    names = registered()
    assert "floor" in names
    assert len(names) >= 5, f"only {len(names)} adapters discovered: {sorted(names)}"


def test_floor_runs(built):
    res = results(built["out"], "floor")
    assert res["binary"] is True
    assert res["n_positives"] == built["n_pos"]
    assert res["n_negatives"] == built["n_neg"]
    assert 0.0 <= res["mean_recall"] <= 1.0


def test_scores_are_saved_for_every_document(built):
    """Without them no operating point can be re-derived, which is half of what ships."""
    f = sorted((built["out"] / "results").glob("graded-*.scores.jsonl"))[-1]
    ids = [json.loads(line)["id"] for line in f.open(encoding="utf-8")]
    assert len(ids) == built["n_pos"] + built["n_neg"]
    assert len(set(ids)) == len(ids)


# --------------------------------------------------------------------------- the numbers cohere

def test_curve_passes_through_the_headline_point(built):
    """REGRESSION. The headline moved to one corpus-wide threshold and the curve kept cutting per
    carrier, so a page showed two protocols at once -- up to 8 points apart -- and nothing looked
    wrong. The curve and the row it sits under must be the same measurement."""
    res = results(built["out"], "graded")
    nearest = min(res["curve"], key=lambda p: abs(p["fpr"] - res["fpr_pooled"]))
    assert nearest["recall"] == pytest.approx(res["mean_recall"], abs=1e-9), (
        f"curve says {nearest['recall']:.4f} at FPR {nearest['fpr']:.5f}, "
        f"headline says {res['mean_recall']:.4f} at {res['fpr_pooled']:.5f}")


def test_recall_rises_with_the_budget(built):
    res = results(built["out"], "graded")
    pts = sorted(res["points"].items(), key=lambda kv: float(kv[0]))
    recalls = [p["mean_recall"] for _, p in pts]
    assert recalls == sorted(recalls), f"recall fell as the budget grew: {recalls}"


def test_every_cell_is_reported(built):
    res = results(built["out"], "graded")
    assert res["n_cells"] == len(corpus.FAMILIES) * len(corpus.ACTIONS)
    assert sum(c["n"] for c in res["cells"].values()) == built["n_pos"]


# --------------------------------------------------------------------------- the pages render

def test_reports_and_index(built):
    reports = built["out"] / "reports"
    assert (reports / "comparison-all.md").is_file()
    index = (reports / "README.md").read_text(encoding="utf-8")
    for detector in ("floor", "graded"):
        assert f"`{detector}`" in index, f"{detector} missing from the report index"
    # Every page the index links to has to exist, or the index is worse than none.
    import re
    for link in re.findall(r"\]\(([^)]+\.md)\)", index):
        assert (reports / link).is_file(), f"index links to a missing page: {link}"


def test_figures_are_valid_xml(built):
    """They are loaded through <img>, so a browser parses them as XML: one bad attribute and the
    figure silently becomes a broken-image icon."""
    import xml.etree.ElementTree as ET
    figures = sorted((built["out"] / "reports" / "figures").glob("*.svg"))
    assert figures, "no figures were written"
    for f in figures:
        ET.fromstring(f.read_text(encoding="utf-8"))


# --------------------------------------------------------------------------- it refuses honestly

def test_missing_corpus_is_an_error_not_a_download(tmp_path):
    """An explicit path that has no corpus must fail loudly. Fetching a different corpus into a
    different place and measuring against it answers a question nobody asked."""
    r = run(["quadrat.run", "--detector", "floor"], tmp_path / "absent", tmp_path / "out",
            check=False)
    assert r.returncode != 0
    assert "corpus incomplete" in (r.stdout + r.stderr)


# --------------------------------------------------------------------------- the cell ramp

def test_ramp_sign_matches_the_claim():
    """Above the hinge the cell greens, below it it warms, and nothing crosses over.

    The grid's colour IS a claim -- held or open -- so a step on the wrong side of the hinge would
    state the opposite of the number written inside it."""
    from quadrat import figures as fg
    for theme in ("light", "dark"):
        card = dict(zip(fg.TOKENS, fg.PALETTES[theme]))["card"]
        # DIRECTION, NOT DISTANCE. A step is a light tint of its hue, so in plain RGB it is far
        # from the deep endpoint it came from and can sit nearer the other one -- which says
        # nothing. What the claim rests on is which way the tile departs from the page: the
        # chroma offset from `card` has to point along its own end and away from the other.
        pal = dict(zip(fg.TOKENS, fg.PALETTES[theme]))
        chroma = lambda h: fg._lin2oklab(*fg._hex2lin(h))[1:]
        c0 = chroma(card)
        off = lambda h: tuple(x - y for x, y in zip(chroma(h), c0))
        dot = lambda u, v: u[0] * v[0] + u[1] * v[1]
        for r, end in ((0.0, "heat_lo"), (0.25, "heat_lo"), (0.75, "heat"), (1.0, "heat")):
            other = "heat" if end == "heat_lo" else "heat_lo"
            step = off(fg.heat_hex(r, theme))
            assert dot(step, off(pal[end])) > 0, f"{theme} {r} does not lean to {end}"
            assert dot(step, off(pal[other])) < 0, f"{theme} {r} leans towards {other}"
        assert fg.heat_hex(fg.HEAT_MID, theme) == card, "the hinge is not the page background"


def test_ramp_keeps_the_numeral_readable():
    """4.5:1 at every step of both ends, in both themes -- the constraint HEAT_DEPTH is set by."""
    from quadrat import figures as fg
    for theme in ("light", "dark"):
        assert fg._contrast_floor(theme) >= 4.5, (
            f"{theme}: cell numerals fall to {fg._contrast_floor(theme):.2f}:1 on the ramp")
