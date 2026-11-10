"""Re-measure trained runs. The test half of a task.

Training writes a checkpoint and stops. Measuring is separate because the
estimators want incompatible batch sizes:

    sampled counting   needs B far above the effective alphabet 2^H(K), which
                       grows with the number of cells. Cheap per sample, but
                       the budget explodes with C.
    KT bounds          cost O(B^2 * R), linear in the array size but QUADRATIC
                       in B. Happy with a modest batch, unhappy with a huge one.

They fail in opposite directions, which is what makes running both a real
cross-check rather than two views of the same number. Folding either into
training would privilege it and force a retrain to change it.

Each pass writes test_<label>.json into every run directory, so the estimators
sit side by side and analysis.load merges them into one row.
"""
import torch

from analysis.load import load_model
from train.io import find_sweeps, is_finished
from train.runner import Runner


def measure_run(run_dir, measurements, batch_size, eval_chunk_size=None,
                device=None, label="results"):
    """Measure one finished run and write test_<label>.json beside its config."""
    from train.io import RunLogger, load_config

    config = load_config(run_dir)
    runner = Runner(config, logger=None, device=device)
    runner.build(verbose=False)
    env, physics, receptor_indices, readout = load_model(run_dir, device=runner.device)
    # Swap the trained weights into the runner's own components, so evaluate()
    # uses the loss and grouping it built rather than a second construction.
    runner.env.load_state_dict(env.state_dict())
    runner.physics.load_state_dict(physics.state_dict())
    runner.physics.temperature = config.temperature
    if runner.readout is not None and readout is not None:
        runner.readout.load_state_dict(readout.state_dict())
    runner.env.bind_receptors(runner.receptor_indices)

    with torch.no_grad():
        results = runner.evaluate(measurements, batch_size, eval_chunk_size)

    # Provenance, not a measurement: how many samples THIS pass used. Two passes
    # legitimately differ here, so scope it to the label. Keys that are genuine
    # measurements stay unscoped, and load_run raises if two passes claim one.
    for key in ("response_evaluation_samples",):
        if key in results:
            results[f"{label}_samples"] = results.pop(key)
    results[f"{label}_batch_size"] = [batch_size]
    RunLogger(run_dir).save_results(results, label)
    return results


def measure_sweep(sweep_root, measurements, batch_size, label,
                  eval_chunk_size=None, device=None, overwrite=False):
    """Measure every finished run under one sweep root."""
    from pathlib import Path

    root = Path(sweep_root)
    done = 0
    for path in sorted(root.rglob("config.json")):
        run_dir = path.parent
        if not is_finished(run_dir):
            continue                       # interrupted: no checkpoint to measure
        if not overwrite and (run_dir / f"test_{label}.json").is_file():
            continue
        measure_run(run_dir, measurements, batch_size, eval_chunk_size, device, label)
        done += 1
        print(f"  [{done}] {run_dir.relative_to(root)}")
    return done


def measure_goal(data_root, prefixes, measurements, batch_size, label,
                 eval_chunk_size=None, device=None, overwrite=False):
    """Measure every sweep under data_root whose name starts with one of `prefixes`.

    Raises when a prefix matches nothing: a silent zero would look like a
    finished measurement pass that simply found nothing to do.
    """
    from pathlib import Path

    root = Path(data_root)
    total = 0
    for prefix in prefixes:
        sweeps = [p for p in find_sweeps(root) if p.name.startswith(prefix)]
        if not sweeps:
            raise FileNotFoundError(
                f"No sweep under {root} starting with {prefix!r}. Train it first.")
        for sweep_root in sweeps:
            print(f"\n{sweep_root.name}  ->  test_{label}.json  (B = {batch_size:,})")
            total += measure_sweep(sweep_root, measurements, batch_size, label,
                                   eval_chunk_size, device, overwrite)
    print(f"\nmeasured {total} run(s) as {label!r}")
    return total
