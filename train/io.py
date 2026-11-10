"""Writing a run to disk, and reading it back.

Layout:

    <data_root>/<goal>/<sweep_name>_<timestamp>/
        sweep_config.json         the whole sweep: every run's config
        .state                    running | complete | failed | interrupted
        <axis>_<value>/.../run_<timestamp>/
            config.json           this run, fully resolved
            stats.csv             training history
            best_model.pt         env + physics + receptor_indices
            test_<label>.json     one per measurement pass

Training writes config.json, stats.csv and best_model.pt. It does NOT measure:
a test script does that afterwards and writes test_<label>.json, one file per
estimator, because different estimators need very different batch sizes.

So a run is FINISHED when it has a checkpoint, not when it has results. A
finished run with no test_*.json has simply not been measured yet.

The directory tree is the only index. Nothing caches it, so nothing can go
stale against it.
"""
import csv
import json
import os
from dataclasses import fields
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from train.config import Config

# Fields whose values are sequences: they cannot go in a directory name.
_UNNAMEABLE = {"receptor_indices", "cell_gene_sets", "cell_receptors",
               "measurement_fns", "kernel_params", "tags",
               "cell_size_pmf", "family_weights", "conc_mean", "conc_std"}


class _Encoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, torch.Tensor):
            return obj.cpu().tolist()
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        return super().default(obj)


def varying_fields(runs):
    """Field names that differ across the sweep. These name the directories."""
    if len(runs) < 2:
        return set()
    return {f.name for f in fields(Config)
            if len({json.dumps(getattr(r, f.name), sort_keys=True, cls=_Encoder)
                    for r in runs}) > 1}


def _run_rel_path(config, axes, run_timestamp):
    """One run's path inside the sweep: one directory per varying scalar axis."""
    parts = []
    for name in sorted(axes):
        if name == "cell_receptors" and config.cell_receptors is not None:
            counts = [len(cell) for cell in config.cell_receptors]
            label = (str(counts[0]) if counts and len(set(counts)) == 1
                     else "-".join(map(str, counts)))
            parts.append(f"receptors_per_cell_{label}")
        elif name not in _UNNAMEABLE:
            parts.append(f"{name}_{getattr(config, name)}")
    parts.append(f"run_{run_timestamp}")
    return os.path.join(*parts)


class RunLogger:
    """Writes one run's directory."""

    def __init__(self, run_dir):
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)

    def save_config(self, config, run_timestamp=""):
        d = config.to_dict()
        if run_timestamp:
            d["run_timestamp"] = run_timestamp
        (self.run_dir / "config.json").write_text(
            json.dumps(d, indent=4, cls=_Encoder))

    def save_stats(self, epoch, stats):
        stats = dict(stats, epoch=epoch)
        path = self.run_dir / "stats.csv"
        exists = path.is_file()
        with path.open("a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(stats))
            if not exists:
                writer.writeheader()
            writer.writerow(stats)

    def save_results(self, results, label):
        """One measurement pass. `label` names the estimator, not the run."""
        if not label or "/" in label:
            raise ValueError(f"label must be a plain name, got {label!r}")
        (self.run_dir / f"test_{label}.json").write_text(
            json.dumps(results, indent=4, cls=_Encoder))

    def save_checkpoint(self, epoch, env, physics, receptor_indices, readout=None):
        state = {
            "epoch": epoch,
            "env_state": env.state_dict(),
            "physics_state": physics.state_dict(),
            "receptor_indices": receptor_indices.cpu(),
        }
        if readout is not None:
            state["readout_state"] = {k: v.cpu() for k, v in readout.state_dict().items()}
        torch.save(state, self.run_dir / "best_model.pt")


class SweepLogger:
    """Creates the sweep directory and vends one RunLogger per run."""

    def __init__(self, runs, name, data_root):
        # Second resolution keeps directory names readable, but two sweeps
        # launched inside one second must NOT land in the same directory: their
        # runs would later be read as one sweep. Suffix instead of merging.
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.root = Path(data_root) / f"{name}_{stamp}"
        attempt = 1
        while self.root.exists():
            attempt += 1
            self.root = Path(data_root) / f"{name}_{stamp}_{attempt}"
        self.root.mkdir(parents=True)
        self.axes = varying_fields(runs)
        (self.root / "sweep_config.json").write_text(
            json.dumps([r.to_dict() for r in runs], indent=4, cls=_Encoder))
        self.set_state("running")

    def set_state(self, state):
        """Machine-owned execution state, written atomically."""
        if state not in {"running", "complete", "failed", "interrupted"}:
            raise ValueError(f"Unknown execution state: {state}")
        tmp = self.root / ".state.tmp"
        tmp.write_text(state + "\n")
        tmp.replace(self.root / ".state")

    def run_logger(self, config):
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        logger = RunLogger(self.root / _run_rel_path(config, self.axes, stamp))
        logger.save_config(config, stamp)
        return logger


# --------------------------------------------------------------------------
# Reading
# --------------------------------------------------------------------------

def load_config(run_dir):
    """Just the Config. Reads no results, so a measuring pass is never blocked
    by what a previous pass wrote."""
    data = json.loads((Path(run_dir) / "config.json").read_text())
    return Config.from_dict({k: v for k, v in data.items() if k != "run_timestamp"})


def load_run(run_dir):
    """(Config, stats DataFrame, results dict) for one run directory.

    Results merge every test_<label>.json present. Two estimators measuring the
    same model produce differently named keys, so they sit side by side and a
    figure can plot one against the other. A key written by two passes raises
    rather than letting the last file read win silently.
    """
    run_dir = Path(run_dir)
    config = load_config(run_dir)
    stats = (pd.read_csv(run_dir / "stats.csv")
             if (run_dir / "stats.csv").is_file() else pd.DataFrame())
    results, source = {}, {}
    for path in sorted(run_dir.glob("test_*.json")):
        label = path.stem[len("test_"):]
        for key, value in json.loads(path.read_text()).items():
            if key in results and results[key] != value:
                raise ValueError(
                    f"{run_dir}: {key!r} differs between test_{source[key]}.json "
                    f"and test_{label}.json. Two passes cannot claim the same "
                    f"measurement; give one of them its own key.")
            results[key], source[key] = value, label
    return config, stats, results


def is_finished(run_dir):
    """A run is finished when it has a checkpoint, measured or not."""
    return (Path(run_dir) / "best_model.pt").is_file()


def iter_runs(sweep_root, measured_only=True):
    """Every finished run under a sweep root, as (Config, stats, results, path).

    A run without a checkpoint was interrupted and must never reach a result.
    With measured_only, runs that no test script has touched are skipped too.
    """
    for path in sorted(Path(sweep_root).rglob("config.json")):
        run_dir = path.parent
        if not is_finished(run_dir):
            continue
        config, stats, results = load_run(run_dir)
        if measured_only and not results:
            continue
        yield config, stats, results, run_dir


def find_sweeps(data_root, prefix=""):
    """Sweep roots under data_root, newest first."""
    roots = [p for p in Path(data_root).iterdir()
             if p.is_dir() and p.name.startswith(prefix)
             and (p / "sweep_config.json").is_file()]
    return sorted(roots, key=lambda p: p.name, reverse=True)
