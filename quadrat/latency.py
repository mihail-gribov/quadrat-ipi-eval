#!/usr/bin/env python3
"""Latency per document -- what a caller waits, not what the corpus costs.

    python3 -m quadrat.latency --detector bastion
    python3 -m quadrat.latency --detector bastion --n 300 --warmup 20

NOT THE SAME NUMBER AS THE PASS. A run over the corpus reports `seconds` and `n_windows`, and
dividing one by the other gives THROUGHPUT at batch 32 on a card that was usually shared -- a
figure about our hardware and our scheduling. What a deployment waits for is one document, alone,
through the detector's own aperture. The two differ by more than a constant: batching is the whole
reason the throughput number looks good, and a service answering one request at a time gets none
of it.

So: batch 1, one document at a time, in corpus order. The aperture stays the detector's own, which
means a long document legitimately costs more -- it is more windows -- and that is the number a
caller sees.

REPORTED AS PERCENTILES, never as a mean. Latency distributions have tails, the tail is what makes
a timeout fire, and an average hides exactly it. p50 says what usually happens; p95 says what the
slowest twentieth of documents costs, which is what a queue is sized against.

COLD START IS SEPARATE. The first calls include weights landing on the card and kernels being
compiled; folding them into the distribution would report a service that restarts before every
request. `--warmup` documents are timed and reported apart rather than dropped silently.

THE CARD IS RECORDED, not assumed to be free. A latency measured while three other passes hold the
GPU is a fact about that moment, so the number of other CUDA processes goes into the record and
the reader can discard a row taken under load.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import subprocess
import time

from .data import DEFAULT_ROOT, SLICES, fingerprint, load
from .detector import REGISTRY, load_detectors

from .paths import RESULTS as OUT


def gpu_processes():
    """How many CUDA processes hold the card right now (this one included)."""
    try:
        r = subprocess.run(["nvidia-smi", "--query-compute-apps=pid",
                            "--format=csv,noheader"], capture_output=True, text=True, timeout=10)
        return len([l for l in r.stdout.splitlines() if l.strip()])
    except Exception:
        return None


def percentile(v, p):
    if not v:
        return 0.0
    v = sorted(v)
    k = (len(v) - 1) * p
    lo, hi = int(k), min(int(k) + 1, len(v) - 1)
    return v[lo] + (v[hi] - v[lo]) * (k - lo)


def measure(det, docs, warmup):
    """[ms per document], plus the warm-up times kept apart."""
    cold, warm = [], []
    for i, d in enumerate(docs):
        t0 = time.perf_counter()
        det.score_documents([d])
        dt = (time.perf_counter() - t0) * 1000
        (cold if i < warmup else warm).append(dt)
    return warm, cold


def main():
    ap = argparse.ArgumentParser(description="per-document latency, batch 1")
    ap.add_argument("--detector", required=True)
    ap.add_argument("--n", type=int, default=200, help="documents timed after warm-up")
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--slice", default="all", choices=sorted(SLICES))
    ap.add_argument("--data", default=str(DEFAULT_ROOT))
    ap.add_argument("--adapters", action="append", default=[])
    a = ap.parse_args()

    load_detectors(a.adapters)
    if a.detector not in REGISTRY:
        raise SystemExit(f"no such detector: {a.detector}")
    cls, version = REGISTRY[a.detector]
    det = cls()
    det.settings = {"latency": True}          # judges refuse to ship raw text without a decision
    det.setup()

    pos, neg = load(a.data, a.slice)
    # A MIXED SAMPLE, alternating injected and clean, because latency follows document LENGTH and
    # the two pools differ in it. Taking the first N of one pool would report the latency of that
    # pool's typical length and call it the detector's.
    docs = [d for pair in zip(pos, neg) for d in pair][:a.n + a.warmup]
    warm, cold = measure(det, docs, a.warmup)
    det.teardown()

    chars = [len(d.text) for d in docs[a.warmup:]]
    rec = {
        "detector": a.detector, "version": version, "notes": det.notes,
        "n": len(warm), "warmup": len(cold),
        "p50_ms": round(percentile(warm, 0.50), 1),
        "p95_ms": round(percentile(warm, 0.95), 1),
        "p99_ms": round(percentile(warm, 0.99), 1),
        "min_ms": round(min(warm), 1) if warm else 0,
        "max_ms": round(max(warm), 1) if warm else 0,
        "cold_first_ms": round(cold[0], 1) if cold else None,
        "median_chars": int(percentile(chars, 0.5)) if chars else 0,
        "gpu_processes": gpu_processes(),
        "aperture": det.aperture(),
        "dataset": fingerprint(a.data),
        "run_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"latency-{a.detector}.json").write_text(
        json.dumps(rec, ensure_ascii=False, indent=1), encoding="utf-8")

    load_note = ("" if rec["gpu_processes"] in (None, 1)
                 else f"  ⚠ {rec['gpu_processes']} processes share the card -- this number is under load")
    print(f"\n=== latency {a.detector} · batch 1 · {rec['n']} documents ===")
    print(f"  p50 {rec['p50_ms']:.1f} ms · p95 {rec['p95_ms']:.1f} · p99 {rec['p99_ms']:.1f} "
          f"· range {rec['min_ms']:.0f}-{rec['max_ms']:.0f}")
    print(f"  first call (cold) {rec['cold_first_ms']} ms · median document "
          f"{rec['median_chars']} chars")
    if load_note:
        print(load_note)
    print(f"-> {OUT / f'latency-{a.detector}.json'}")


if __name__ == "__main__":
    main()
