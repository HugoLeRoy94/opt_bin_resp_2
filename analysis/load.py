"""Finished runs on disk -> one tidy DataFrame.

    points = across_replicates(runs("fig2"))

`runs()` gives one row per finished run: every scalar config field as a column,
every recorded measurement as a column, plus `sweep` and `run_dir`. Nothing is
task-specific, so a task's figure.py selects and labels what it needs rather
than each one re-implementing discovery.

Replicates are INDEPENDENT WORLDS. Their spread is the quantity that belongs on
an error bar, and with only a handful of them a standard error would be fiction,
so `across_replicates` reports mean with min/max. Nothing here writes a file or
prints on the happy path; it raises when the answer would otherwise be wrong.
"""
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from train.config import Config
from train.io import find_sweeps, iter_runs, load_run

DATA_ROOT = Path(os.environ.get("OCTOPUS_DATA_ROOT",
                                Path(__file__).resolve().parents[1] / "data"))

# Sequence-valued config fields. They cannot be a DataFrame cell without turning
# the column into objects that groupby and unique choke on, so they are
# summarised instead; the raw values stay reachable through run_dir.
_SEQUENCE_FIELDS = {
    "receptor_indices":      lambda v: len(v) if v else 0,
    "cell_gene_sets":        lambda v: len(v) if v else 0,
    "cell_receptors":        lambda v: len(v) if v else 0,
    "cell_size_pmf":         lambda v: len(v) if v else 0,
    "measurement_fns":       lambda v: ",".join(v) if v else "",
    "final_measurement_fns": lambda v: ",".join(v) if v else "",
    "family_weights":        lambda v: ",".join(f"{x:g}" for x in v) if v else "",
    "kernel_params":         lambda v: v[0] if v else np.nan,
}
_RENAMED = {"receptor_indices": "n_receptors_resolved",
            "cell_gene_sets": "n_cells_resolved",
            "cell_receptors": "n_cell_repertoires",
            "kernel_params": "kernel_lambda"}


def _row(config: Config, results: dict):
    row = {}
    for name, value in config.to_dict().items():
        if name in _SEQUENCE_FIELDS:
            row[_RENAMED.get(name, name)] = _SEQUENCE_FIELDS[name](value)
        elif isinstance(value, (list, tuple)):
            # conc_mean / conc_std may be per-ligand tuples; a scalar stays scalar.
            row[name] = value[0] if len(value) == 1 else ",".join(map(str, value))
        else:
            row[name] = value
    # Same rule the old index used: an unset n_receptors means Config fell back
    # to one homomer per gene. Note a cascading array with R == n_genes is the
    # SAME array, so this labels intent, not structure.
    row["receptor_type"] = "homomer" if config.n_receptors is None else "heteromer"
    # Measurements are stored as one-element lists so every consumer has one
    # shape; the mean of a one-element list is that element.
    for key, value in results.items():
        row[key] = float(np.mean(value)) if isinstance(value, list) else value
    return row


def runs(goal, data_root=None, sweeps=None, latest_only=False):
    """One row per finished run under <data_root>/<goal>/.

    sweeps      restrict to these sweep directory names (or prefixes).
    latest_only keep only the newest sweep per prefix, where a prefix is the
                sweep name with its timestamp stripped.
    """
    root = Path(data_root or DATA_ROOT) / goal
    if not root.is_dir():
        raise FileNotFoundError(
            f"No goal directory {root}. Run the sweep first, or pull it with "
            f"ops/manage_data.py sync {goal}.")

    roots = find_sweeps(root)
    if sweeps is not None:
        wanted = tuple(sweeps)
        roots = [p for p in roots if p.name.startswith(wanted)]
    if latest_only:
        seen, kept = set(), []
        for path in roots:                       # find_sweeps is newest first
            prefix = path.name.rsplit("_", 2)[0]
            if prefix not in seen:
                seen.add(prefix)
                kept.append(path)
        roots = kept
    if not roots:
        raise FileNotFoundError(f"No sweeps under {root} matching {sweeps!r}.")

    rows = []
    for sweep_root in roots:
        for config, _stats, results, run_dir in iter_runs(sweep_root):
            rows.append(dict(_row(config, results),
                             sweep=sweep_root.name, run_dir=str(run_dir)))
    if not rows:
        raise RuntimeError(
            f"{len(roots)} sweep(s) under {root} but no FINISHED run in any of "
            f"them. Runs without test_results.json were interrupted.")
    return pd.DataFrame(rows)


def epochs(frame):
    """Training history of the given runs, long form, one row per logged epoch."""
    from train.io import load_run
    parts = []
    for run_dir in frame["run_dir"]:
        _config, stats, _results = load_run(run_dir)
        if not stats.empty:
            parts.append(stats.assign(run_dir=run_dir))
    if not parts:
        raise RuntimeError("None of these runs wrote a stats.csv.")
    return frame.merge(pd.concat(parts, ignore_index=True), on="run_dir",
                       suffixes=("", "_epoch"))


def across_replicates(points, keys, spread="std", metrics=None):
    """Collapse replicates to a mean and a band, one row per design point.

    Two uncertainties exist here and they must not be confused.

      Within one run, the KT lower and upper bounds BRACKET the true H(s). That
      is an estimator guarantee about a single trained model, not noise, and it
      does not narrow when you run more.

      Across runs, independently drawn environments give genuine variability.
      That is what this summarises, and it is what an error band should mean.

    Replicates must be independently drawn WORLDS, not repeats inside one world.
    Resampling within a single chemistry gives a band that understates how far a
    conclusion would move in a different one.

    `keys` define a design point. Every numeric column not in `keys` is
    aggregated; each metric m gains "m lo" and "m hi". Column "n" is how many
    worlds went into the row, and "spread" records the convention used, so a
    frame cannot later be read under the wrong one.

    spread:
      "std"  mean -/+ the standard deviation across worlds: how much chemistries
             differ. The right bar for "does this hold across environments".
      "sem"  mean -/+ std/sqrt(n): how well the MEAN is known. Only when the
             claim is about the mean rather than about the spread.

    With n = 1 the band is NaN, not zero width. A single world supports no band,
    and saying so is better than drawing a line that looks certain.

    Raises on a missing key. Silently collapsing a dimension you meant to keep is
    exactly how a wrong number reaches a figure.
    """
    if spread not in ("std", "sem"):
        raise ValueError(
            f"spread must be 'std' or 'sem', not {spread!r}. min/max was removed: "
            f"its expected width GROWS with n, so bands were not comparable "
            f"between points backed by different numbers of worlds.")
    keys = list(keys)
    missing = [k for k in keys if k not in points.columns]
    if missing:
        raise KeyError(f"Not columns of this frame: {missing}. "
                       f"Available: {sorted(points.columns)}")
    if metrics is None:
        metrics = [c for c in points.columns
                   if c not in keys and pd.api.types.is_numeric_dtype(points[c])]

    grouped = points.groupby(keys, sort=True)
    out = grouped[metrics].mean()
    counts = grouped.size()
    # ddof=1 is the sample std. One world gives NaN, which must stay NaN.
    deviation = grouped[metrics].std(ddof=1)
    if spread == "sem":
        deviation = deviation.div(np.sqrt(counts), axis=0)
    for m in metrics:
        out[f"{m} lo"] = out[m] - deviation[m]
        out[f"{m} hi"] = out[m] + deviation[m]
    out["n"] = counts
    out["spread"] = spread
    return out.reset_index()


def load_model(run_dir, device="cpu"):
    """Rebuild a finished run's model and load its trained weights.

    Returns (env, physics, receptor_indices, readout); readout is None outside
    cell mode. Everything is on `device` with gradients disabled, ready to drive.

    Construction goes through Runner._build so there is exactly ONE place that
    knows how a Config becomes a model. A second copy here would drift the first
    time the environment gains a parameter.
    """
    from train.runner import Runner

    run_dir = Path(run_dir)
    config, _stats, _results = load_run(run_dir)
    runner = Runner(config, logger=None, device=device)
    env, physics, _loss, _opt, receptor_indices = runner._build(verbose=False)

    state = torch.load(run_dir / "best_model.pt", map_location=device,
                       weights_only=False)
    env.load_state_dict(state["env_state"])
    physics.load_state_dict(state["physics_state"])
    if runner.readout is not None and "readout_state" in state:
        runner.readout.load_state_dict(state["readout_state"])

    # The endpoint temperature is what every measurement used, and it is NOT in
    # the state dict: it is a plain attribute the schedule annealed.
    physics.temperature = config.temperature
    env.eval(), physics.eval()
    for p in list(env.parameters()) + list(physics.parameters()):
        p.requires_grad_(False)
    return env, physics, receptor_indices, runner.readout
