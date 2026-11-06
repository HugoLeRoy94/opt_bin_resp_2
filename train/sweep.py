"""Run a list of Configs and record the sweep's execution state.

Runs are independent: nothing is carried from one to the next. That is a
deliberate property, not an omission. If step k started from step k-1's trained
environment, the spread across a sweep would be correlated through a shared
starting point and could not be read as independent replicates.
"""
import gc

import torch
from tqdm import tqdm

from train.io import SweepLogger
from train.runner import Runner


class Sweep:
    """Execute `runs`, writing into <data_root>/<name>_<timestamp>/."""

    def __init__(self, runs, name, data_root, device=None):
        if not runs:
            raise ValueError("Nothing to run: sweep() returned an empty list.")
        self.runs = list(runs)
        self.name = name
        self.data_root = data_root
        self.device = device

    def execute(self):
        logger = SweepLogger(self.runs, self.name, self.data_root)
        print(f"\nSweep: {logger.root}\nRuns:  {len(self.runs)}\n")
        try:
            self._execute(logger)
        except KeyboardInterrupt:
            logger.set_state("interrupted")
            raise
        except Exception:
            logger.set_state("failed")
            raise
        logger.set_state("complete")
        return logger.root

    def _execute(self, logger):
        axes = sorted(logger.axes)
        with tqdm(total=len(self.runs), desc="sweep", dynamic_ncols=True) as bar:
            for config in self.runs:
                label = " | ".join(
                    f"{k}={getattr(config, k)}" for k in axes
                    if not isinstance(getattr(config, k), (list, tuple)))
                if label:
                    tqdm.write(f"--- {label} ---")
                Runner(config, logger.run_logger(config), self.device).run()
                # Release this run's tensors before the next one allocates.
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                bar.update(1)
