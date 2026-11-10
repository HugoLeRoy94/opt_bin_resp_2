"""Record reference numbers from the OLD repo. Run once, before porting.

    python3 test/golden/make_golden.py

Writes receptor.json and cell.json next to this file. Everything the port does
afterwards is checked against them by test/test_golden.py.
"""
import json
import sys
import tempfile
from pathlib import Path

OLD = Path("/mnt/hcleroy/PostDoc2/octopus_smelling/opt_bin_resp")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(OLD))
sys.path.insert(0, str(HERE))

import torch
from configs import (CELL, CELL_MEASURE, PINNED, RECEPTOR,
                     RECEPTOR_MEASURE, SEED)
from src.config import SingleRunConfig
from src.IO import ExperimentLogger
from src.run import SimulationRunner


def record(name, fields, measure):
    """The old repo measures inside run(), so fold the measuring step back in."""
    torch.manual_seed(SEED)
    cfg = SingleRunConfig(**{**fields, **PINNED,
                             "final_measurement_fns": measure["measurements"],
                             "final_test_batch_size": measure["batch_size"]})
    with tempfile.TemporaryDirectory() as tmp:
        SimulationRunner(cfg, ExperimentLogger(tmp)).run()
        result = json.loads((Path(tmp) / "test_results.json").read_text())
    (HERE / f"{name}.json").write_text(json.dumps(result, indent=2, sort_keys=True))
    print(f"{name}: " + ", ".join(f"{k}={v}" for k, v in sorted(result.items())))


if __name__ == "__main__":
    record("receptor", RECEPTOR, RECEPTOR_MEASURE)
    record("cell", CELL, CELL_MEASURE)
