#!/usr/bin/env python3
"""Curate sweeps, and mirror them from the cluster.

    python ops/manage_data.py curate [goal ...]
    python ops/manage_data.py sync   [goal ...] [--yes]

The cluster is authoritative for raw data. curation.csv is authoritative
for the human keep/delete decision. Analyses read the run directories
directly, so there is no index that could fall out of step with either.

`curate` walks the sweeps nobody has judged yet and asks. `sync` applies the
deletions on both machines, then mirrors the cluster down.
"""
import argparse
import os
import re
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

from analysis import curation

ROOT = Path(__file__).resolve().parents[1]
LOCAL_DATA = Path(os.environ.get("OCTOPUS_DATA_ROOT", ROOT / "data")).resolve()
SERVER = os.environ.get("OCTOPUS_DATA_SERVER", "leroy@10.187.172.7")
REMOTE_DATA = os.environ.get("OCTOPUS_REMOTE_DATA_ROOT",
                             "/storage/leroy/data").rstrip("/")

_COMPONENT = re.compile(r"^[A-Za-z0-9_.-]+$")
_TIMESTAMP = re.compile(r"\d{8}_\d{6}")
# ControlMaster reuses one connection, so a sync of twelve goals opens one SSH
# session rather than twelve. BatchMode fails instead of hanging on a prompt.
SSH = ["-o", "BatchMode=yes", "-o", "ControlMaster=auto",
       "-o", "ControlPersist=60", "-o", "ControlPath=/tmp/octopus-ssh-%C"]
RSYNC = ["-az", "--delete"]


def _component(value, what):
    """Refuse anything that is not a plain name. These reach `rm -rf` and rsync."""
    if not _COMPONENT.match(value):
        raise ValueError(f"Unsafe {what}: {value!r}")
    return value


def _sweep_path(relative):
    goal, _, sweep = relative.partition("/")
    return f"{_component(goal, 'goal')}/{_component(sweep, 'sweep')}"


def _sweeps(goals):
    found = []
    for goal in goals:
        root = LOCAL_DATA / goal
        if root.is_dir():
            found += [c for c in root.iterdir()
                      if c.is_dir() and _TIMESTAMP.search(c.name)]
    return sorted(found)


def _goals(requested, decisions):
    if requested and requested != ["all"]:
        return sorted({_component(g, "goal") for g in requested})
    local = {p.name for p in LOCAL_DATA.iterdir() if p.is_dir()} if LOCAL_DATA.is_dir() else set()
    return sorted(local | {k.split("/", 1)[0] for k in decisions})


def _size(path):
    total = sum(p.stat().st_size for p in path.rglob("*") if p.is_file())
    value = float(total)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if value < 1024 or unit == "TiB":
            return f"{value:.1f} {unit}"
        value /= 1024


def _summary(sweep):
    """(execution state, trained, measured, planned) for one sweep.

    Trained and measured are counted separately because they are separate
    steps: a run is finished when it has a checkpoint, and measured only once a
    test script has written a test_*.json beside it.
    """
    configs = list(sweep.rglob("config.json"))
    trained = sum((p.parent / "best_model.pt").is_file() for p in configs)
    measured = sum(any(p.parent.glob("test_*.json")) for p in configs)
    marker = sweep / ".state"
    state = marker.read_text().strip() if marker.is_file() else "unknown"
    return state, trained, measured, len(configs)


# ── curate ──────────────────────────────────────────────────────────────────

def curate(goals, show_all):
    decisions = curation.read()
    selected = _goals(goals, decisions)
    sweeps = _sweeps(selected)
    if not show_all:
        sweeps = [s for s in sweeps if f"{s.parent.name}/{s.name}" not in decisions]
    if not sweeps:
        print("No sweeps need review.")
        return

    for sweep in sweeps:
        key = f"{sweep.parent.name}/{sweep.name}"
        state, trained, measured, planned = _summary(sweep)
        current, label = decisions.get(key, ("review", ""))
        print(f"\n{key}")
        print(f"  execution: {state}")
        print(f"  runs:      {trained}/{planned} trained, {measured} measured")
        print(f"  size:      {_size(sweep)}")
        print(f"  curation:  {current}" + (f" - {label}" if label else ""))
        choice = input("  [k]eep  [d]elete  [s]kip  [q]uit: ").strip().lower()
        if choice == "q":
            break
        if choice in ("", "s"):
            continue
        if choice == "k":
            label = input("  short label: ").strip()
            if not label:
                print("  skipped: a kept sweep needs a label saying why")
                continue
            decisions[key] = ("keep", label)
        elif choice == "d":
            decisions[key] = ("delete", "")
        else:
            print("  unknown choice; skipped")
            continue
        curation.write(decisions)
        print(f"  saved: {decisions[key][0]}")


# ── sync ────────────────────────────────────────────────────────────────────

def _ssh(args, check=True):
    return subprocess.run(["ssh", *SSH, SERVER, *args], check=check)


def _remote_exists(path):
    return _ssh(["test", "-e", path], check=False).returncode == 0


def _remote_goals():
    command = (f"find {shlex.quote(REMOTE_DATA)} -mindepth 1 -maxdepth 1 "
               "-type d -printf '%f\\n'")
    out = subprocess.run(["ssh", *SSH, SERVER, command], check=True,
                         capture_output=True, text=True).stdout
    return sorted(_component(line.strip(), "remote goal")
                  for line in out.splitlines() if line.strip())


def _apply_deletions(decisions, goals, assume_yes):
    targets = sorted(k for k, (state, _) in decisions.items()
                     if state == "delete" and k.split("/", 1)[0] in goals)
    if not targets:
        return []
    print("\nSweeps labelled delete (cluster and local):")
    for key in targets:
        print(f"  {key}")
    if not assume_yes:
        if not sys.stdin.isatty():
            raise RuntimeError("Deletion needs an interactive confirmation or --yes.")
        if input("Type 'delete' to continue: ").strip() != "delete":
            raise RuntimeError("Cancelled; nothing was deleted.")

    # Remote first, in one validated call. If SSH fails the local copy is still
    # there and sync stops. Deleting an absent path is harmless and idempotent.
    validated = [_sweep_path(key) for key in targets]
    _ssh(["rm", "-rf", "--", *[f"{REMOTE_DATA}/{rel}" for rel in validated]])
    for rel in validated:
        print(f"  deleted/absent remote: {rel}")
    for key in targets:
        local = LOCAL_DATA / key
        if local.is_dir():
            shutil.rmtree(local)
            print(f"  deleted local:  {key}")
    return targets


def _sync_goal(goal, assume_yes):
    """Mirror one goal down. The cluster is authoritative for raw data.

    --delete makes the local tree a true mirror, so a sweep that exists only
    locally disappears. That is silent data loss for anything never pushed, so
    a dry run names the local-only sweeps and asks before any of it happens.
    """
    remote = f"{REMOTE_DATA}/{goal}"
    if not _remote_exists(remote):
        print(f"skip {goal}: no cluster directory")
        return
    local = LOCAL_DATA / goal
    local.mkdir(parents=True, exist_ok=True)
    print(f"\nsync {goal}")
    transfer = ["-e", "ssh " + " ".join(SSH), f"{SERVER}:{remote}/", f"{local}/"]
    preview = subprocess.run(
        ["rsync", *RSYNC, "--dry-run", "--out-format=%o %n", *transfer],
        check=True, capture_output=True, text=True).stdout
    doomed = sorted({line.split(" ", 1)[1].strip("/").split("/")[0]
                     for line in preview.splitlines()
                     if line.startswith("del. ") and line.count("/") >= 1})
    if doomed:
        print(f"  {len(doomed)} local sweep(s) absent from the cluster will be removed:")
        for name in doomed:
            print(f"    {goal}/{name}")
        if not assume_yes:
            if not sys.stdin.isatty():
                raise RuntimeError(f"Mirroring {goal} would delete local-only data; "
                                   "rerun interactively or pass --yes.")
            if input("  Type 'mirror' to continue: ").strip() != "mirror":
                raise RuntimeError("Cancelled; nothing was deleted.")
    subprocess.run(["rsync", *RSYNC, *transfer], check=True)


def sync(goals, assume_yes):
    decisions = curation.read()
    selected = _goals(goals, decisions)
    if not goals or goals == ["all"]:
        selected = sorted(set(selected) | set(_remote_goals()))

    deleted = _apply_deletions(decisions, selected, assume_yes)
    # A carried-out deletion needs no tombstone: drop the row so later syncs
    # stay quiet. The keep labels remain as the durable catalogue.
    if deleted:
        for key in deleted:
            decisions.pop(key, None)
        curation.write(decisions)

    for goal in selected:
        _sync_goal(goal, assume_yes)

    undecided = [s for s in _sweeps(selected)
                 if f"{s.parent.name}/{s.name}" not in curation.read()]
    print(f"\nSync complete. {len(undecided)} sweep(s) still need review.")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    c = commands.add_parser("curate", help="label sweeps keep or delete")
    c.add_argument("goals", nargs="*", metavar="GOAL")
    c.add_argument("--all", action="store_true", help="revisit already-labelled sweeps")
    s = commands.add_parser("sync", help="apply deletions, then mirror the cluster")
    s.add_argument("goals", nargs="*", metavar="GOAL")
    s.add_argument("--yes", action="store_true", help="skip the confirmations")
    a = parser.parse_args()
    if a.command == "curate":
        curate(a.goals, a.all)
    else:
        sync(a.goals, a.yes)


if __name__ == "__main__":
    main()
