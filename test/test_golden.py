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

from configs import CELL, RECEPTOR, SEED          # noqa: E402
from train.config import Config                   # noqa: E402
from train.io import RunLogger                    # noqa: E402
from train.runner import Runner                   # noqa: E402


def _run(fields):
    torch.manual_seed(SEED)
    with tempfile.TemporaryDirectory() as tmp:
        logger = RunLogger(tmp)
        Runner(Config(**fields), logger, device="cpu").run()
        return json.loads((pathlib.Path(tmp) / "test_results.json").read_text())


@pytest.mark.parametrize("name,fields", [("receptor", RECEPTOR), ("cell", CELL)])
def test_reproduces_golden(name, fields):
    expected = json.loads((HERE / "golden" / f"{name}.json").read_text())
    got = _run(fields)
    assert set(got) == set(expected), (
        f"measurement keys changed: only in new {set(got) - set(expected)}, "
        f"only in golden {set(expected) - set(got)}")
    for key, want in expected.items():
        assert got[key] == pytest.approx(want, rel=1e-6, abs=1e-9), (
            f"{name}.{key}: {got[key]} != {want}")
