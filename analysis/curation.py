"""Which sweeps are worth keeping, and why.

`curation.csv` at the repository root is the human keep/delete decision. It is tracked in git, so
it survives any local data being deleted or re-synced, and it is the ONLY
source of that decision: a sweep directory carries no opinion about its own
worth. Scientific value is independent of whether execution succeeded.

Three states. A sweep is `keep` with a short label saying why, `delete`, or
absent from the file, which means nobody has decided yet. A `keep` without a
label is not a decision and is rejected.

This module READS, and analysis.load surfaces the state as a column so a figure
can restrict itself to kept sweeps. ops/manage_data.py writes through it, so
the tool and the figures cannot disagree about what a row means.
"""
import csv
import os
from pathlib import Path

CURATION_PATH = Path(os.environ.get(
    "OCTOPUS_CURATION_FILE", Path(__file__).resolve().parents[1] / "curation.csv"))
STATES = frozenset({"keep", "delete"})
_CACHE = (None, {})


def read(path=None, strict=False):
    """Parse the registry into {"goal/sweep": (state, label)}.

    strict=True raises on a malformed row and is used before rewriting the
    file. The default skips it, so loading a figure never fails because of an
    unrelated typo three rows down. Cached until the file's mtime changes.
    """
    global _CACHE
    path = Path(path or CURATION_PATH)
    try:
        stamp = (str(path), path.stat().st_mtime_ns)
    except FileNotFoundError:
        stamp = None
    if not strict and stamp is not None and _CACHE[0] == stamp:
        return dict(_CACHE[1])

    catalog = {}
    if stamp is not None:
        with path.open(newline="") as f:
            for row in csv.DictReader(f):
                key = row.get("path", "").strip().strip("/")
                state = row.get("state", "").strip()
                label = row.get("label", "").strip()
                if not key or state not in STATES:
                    if strict:
                        raise ValueError(f"Invalid state {state!r} for {key!r}")
                    continue
                if state == "keep" and not label:
                    if strict:
                        raise ValueError(f"A kept sweep needs a label: {key}")
                    continue
                catalog[key] = (state, label)
    if not strict:
        _CACHE = (stamp, dict(catalog))
    return catalog


def write(decisions, path=None):
    """Rewrite the registry, sorted, so a diff shows the decision not the order."""
    path = Path(path or CURATION_PATH)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["path", "state", "label"])
        for key in sorted(decisions):
            state, label = decisions[key]
            writer.writerow([key, state, label])
    global _CACHE
    _CACHE = (None, {})


def state_of(goal, sweep_name, path=None):
    """(state, label) for one sweep; ("review", "") when nobody has decided."""
    return read(path).get(f"{goal}/{sweep_name}", ("review", ""))
