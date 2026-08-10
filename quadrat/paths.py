#!/usr/bin/env python3
"""Where the harness reads the corpus and where it writes what it produces.

ONE PLACE, because these were four. `run.py`, `compare.py`, `latency.py` and `rescore.py` each
computed their own output directory from `__file__`, which put the measurements INSIDE the
installed package: cloning the tool brought somebody else's results with it, and a run wrote into
the source tree rather than into the working directory. The corpus, the harness and the
measurements are three artifacts with three lifetimes, and only the harness belongs in this
repository.

    QUADRAT_DATA   the corpus: two JSONL files, positives and negatives.   Default: ./data
    QUADRAT_OUT    everything a run produces.                              Default: ./eval-out

Both are read relative to the directory the command was run from, never to this file. `--data` and
`--results` override them per command.

REPORTS SIT BESIDE RESULTS, not inside. `results/` is machine output -- scores, checkpoints, one
JSON per run -- and mixing a directory of rendered pages into it made "delete the stale
measurements" a command that also deleted the reports.
"""
from __future__ import annotations

import os
import pathlib

#: The corpus. `fingerprint()` hashes the two files here, and every result records the result.
DATA_ROOT = pathlib.Path(os.environ.get("QUADRAT_DATA", "data"))

#: Where the corpus comes from when it is not on disk, and which released version to take.
#: PINNED TO A TAG, never to `main`: a Hugging Face repository is a git repository, every release
#: is tagged, and a harness that silently followed the newest state would change what a saved
#: score means between two runs of the same code.
DATASET_REPO = os.environ.get("QUADRAT_REPO", "mihailgribov/quadrat-ipi")
#: BUMPED WITH EACH DATASET RELEASE, and deliberately not `main`. The point of the default is
#: that two people running this command a year apart measure the same documents, and that
#: their numbers are comparable with the table published for this version. Set
#: `QUADRAT_VERSION=main` to follow the newest state instead -- the fingerprint recorded in
#: every result still says which bytes you actually measured.
DATASET_VERSION = os.environ.get("QUADRAT_VERSION", "v1.0.1")

CORPUS_FILES = ("positives.jsonl", "negatives.jsonl")


def corpus(root=None) -> pathlib.Path:
    """The directory holding the corpus, fetching it on first use if it is not there.

    AN EXPLICIT PATH IS NEVER SECOND-GUESSED. If `QUADRAT_DATA` or `--data` names a directory and
    the files are not in it, that is an error worth seeing -- downloading a different corpus into
    a different place, and measuring against it, would answer a question nobody asked. The fetch
    happens only for the default location.
    """
    root = pathlib.Path(root) if root is not None else DATA_ROOT
    if all((root / name).is_file() for name in CORPUS_FILES):
        return root
    if os.environ.get("QUADRAT_DATA") or root != DATA_ROOT:
        missing = [n for n in CORPUS_FILES if not (root / n).is_file()]
        raise SystemExit(f"{root}: corpus incomplete, missing {missing}. "
                         f"Point QUADRAT_DATA at a released version's data/ directory, or unset "
                         f"it to fetch {DATASET_REPO} {DATASET_VERSION} automatically.")
    return fetch()


def fetch() -> pathlib.Path:
    """Download the pinned version from the Hub into its cache and return its `data/`."""
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        raise SystemExit(
            "the corpus is not on disk and huggingface_hub is not installed.\n"
            f"  Either `pip install huggingface_hub` to fetch {DATASET_REPO} "
            f"{DATASET_VERSION},\n"
            "  or download it yourself and point QUADRAT_DATA at its data/ directory.")
    print(f"corpus not found locally -- fetching {DATASET_REPO} {DATASET_VERSION} "
          f"(about 380 MB, once)", flush=True)
    # Only the corpus: a release also carries every detector's saved scores and rendered reports,
    # which a run does not need and which are most of its size.
    try:
        path = snapshot_download(repo_id=DATASET_REPO, repo_type="dataset",
                                 revision=DATASET_VERSION, allow_patterns=["data/*"])
    except Exception as e:
        raise SystemExit(
            f"could not fetch {DATASET_REPO} at {DATASET_VERSION}: {e}\n"
            f"  If that revision no longer exists, this harness is older than the dataset.\n"
            f"  Set QUADRAT_VERSION to a tag listed at "
            f"https://huggingface.co/datasets/{DATASET_REPO}/tags, or QUADRAT_VERSION=main "
            f"to take the newest state.")
    out = pathlib.Path(path) / "data"
    if not all((out / name).is_file() for name in CORPUS_FILES):
        raise SystemExit(f"{DATASET_REPO} {DATASET_VERSION}: no data/ in the downloaded snapshot")
    print(f"corpus ready: {out}", flush=True)
    return out

#: Everything a run produces, under one root so a whole measurement campaign moves as a unit.
OUT_ROOT = pathlib.Path(os.environ.get("QUADRAT_OUT", "eval-out"))
RESULTS = OUT_ROOT / "results"
REPORTS = OUT_ROOT / "reports"
