"""The ported code must reproduce the old code's numbers exactly.

test/golden/*.json were recorded from the pre-refactor repo by
test/golden/make_golden.py. They are the only remaining reference: the 189 MB
of existing runs are not readable by this code, by design.

A refactor that changes a number here is a bug, not a simplification.
"""
import json
import pathlib
import sys
import tempfile

import pytest
import torch

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "golden"))

from configs import (CELL, CELL_MEASURE, RECEPTOR,     # noqa: E402
                     RECEPTOR_MEASURE, SEED)
from train.config import Config                   # noqa: E402
from train.io import RunLogger                    # noqa: E402
from train.runner import Runner                   # noqa: E402


def _run(fields, measure):
    """Train, then measure, which is now two steps rather than one call."""
    torch.manual_seed(SEED)
    with tempfile.TemporaryDirectory() as tmp:
        runner = Runner(Config(**fields), RunLogger(tmp), device="cpu")
        runner.run()
        return runner.evaluate(**measure)


@pytest.mark.parametrize("name,fields,measure", [
    ("receptor", RECEPTOR, RECEPTOR_MEASURE),
    ("cell", CELL, CELL_MEASURE)])
def test_reproduces_golden(name, fields, measure):
    expected = json.loads((HERE / "golden" / f"{name}.json").read_text())
    got = _run(fields, measure)
    assert set(got) == set(expected), (
        f"measurement keys changed: only in new {set(got) - set(expected)}, "
        f"only in golden {set(expected) - set(got)}")
    for key, want in expected.items():
        assert got[key] == pytest.approx(want, rel=1e-6, abs=1e-9), (
            f"{name}.{key}: {got[key]} != {want}")


def test_training_does_not_measure():
    """Training writes a checkpoint and stops.

    Folding a measurement into training would privilege one estimator, and
    changing it would mean retraining. Measuring is tasks/_lib/measure.py.
    """
    torch.manual_seed(SEED)
    with tempfile.TemporaryDirectory() as tmp:
        Runner(Config(**RECEPTOR), RunLogger(tmp), device="cpu").run()
        written = {p.name for p in pathlib.Path(tmp).iterdir()}
    assert {"best_model.pt", "config.json", "stats.csv"} <= written, written
    assert not [n for n in written if n.startswith("test_")], written


def test_evaluate_is_independent_of_the_training_batch():
    """A test script picks its own budget; it is not the config's batch_size."""
    torch.manual_seed(SEED)
    with tempfile.TemporaryDirectory() as tmp:
        runner = Runner(Config(**RECEPTOR), RunLogger(tmp), device="cpu")
        runner.run()
        small = runner.evaluate(("mutual_information_kt",), batch_size=64)
        large = runner.evaluate(("mutual_information_kt",), batch_size=256)
    assert small["response_evaluation_samples"] == [64]
    assert large["response_evaluation_samples"] == [256]


def test_a_chunk_local_measurement_records_what_it_actually_saw():
    """full_array_entropy runs on ONE forward pass, not the whole budget.

    Asking for a huge batch does not make it use one, so the sample count it
    reports must be the number it really saw.
    """
    torch.manual_seed(SEED)
    with tempfile.TemporaryDirectory() as tmp:
        runner = Runner(Config(**dict(RECEPTOR, eval_chunk_size=32)),
                        RunLogger(tmp), device="cpu")
        runner.run()
        out = runner.evaluate(("full_array_entropy",), batch_size=4096)
    assert out["response_evaluation_samples"] == [32], out["response_evaluation_samples"]
