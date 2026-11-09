"""The write-then-read path, and the two ways it can be silently wrong.

Both cases here were real bugs found by running a sweep, not by reading:
two sweeps launched inside one second merged into one directory, and an
unset optional sequence field crashed the loader.
"""
import pathlib
import sys
import tempfile

import pytest

from analysis.load import across_replicates, runs
from train.config import Config, sweep
from train.io import SweepLogger

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "golden"))
from configs import RECEPTOR                      # noqa: E402

RESULT = {"full_array_entropy": [1.5], "mutual_information_kt": [0.5]}


def _write_sweep(data_root, name, n_genes_values, offset=0.0):
    """One sweep directory, written by the real writers, with fake results."""
    configs = sweep(dict(RECEPTOR), n_genes=n_genes_values)
    logger = SweepLogger(configs, name, data_root)
    for i, config in enumerate(configs):
        run_logger = logger.run_logger(config)
        run_logger.save_results({k: [v[0] + offset + i] for k, v in RESULT.items()})
    logger.set_state("complete")
    return logger.root


def test_two_sweeps_in_one_second_do_not_merge():
    """Second-resolution timestamps collide. Merging would read them as one sweep."""
    with tempfile.TemporaryDirectory() as tmp:
        root = pathlib.Path(tmp) / "goal"
        a = _write_sweep(root, "demo", [3, 4])
        b = _write_sweep(root, "demo", [3, 4])
        assert a != b, "two sweeps landed in the same directory"
        frame = runs("goal", data_root=tmp)
        assert len(frame) == 4
        assert frame["sweep"].nunique() == 2


def test_unfinished_runs_are_skipped():
    """A run without test_results.json was interrupted and must not be averaged."""
    with tempfile.TemporaryDirectory() as tmp:
        root = pathlib.Path(tmp) / "goal"
        configs = sweep(dict(RECEPTOR), n_genes=[3, 4, 5])
        logger = SweepLogger(configs, "demo", root)
        for config in configs[:2]:
            logger.run_logger(config).save_results(RESULT)
        logger.run_logger(configs[2])             # config.json only, no results
        assert len(runs("goal", data_root=tmp)) == 2


def test_optional_sequence_fields_load_when_unset():
    """cell_gene_sets and friends are None in receptor mode; that is not an error."""
    with tempfile.TemporaryDirectory() as tmp:
        _write_sweep(pathlib.Path(tmp) / "goal", "demo", [3, 4])
        frame = runs("goal", data_root=tmp)
        assert frame["n_cells_resolved"].tolist() == [0, 0]
        # RECEPTOR asks for n_receptors=6, so both runs resolve to 6 whatever
        # n_genes is; the sweep axis changes the POOL the array is drawn from.
        assert frame["n_receptors_resolved"].tolist() == [6, 6]
        assert sorted(frame["n_genes"]) == [3, 4]


def test_across_replicates_refuses_a_missing_key():
    """Collapsing a dimension you meant to keep is how a wrong number ships."""
    with tempfile.TemporaryDirectory() as tmp:
        _write_sweep(pathlib.Path(tmp) / "goal", "demo", [3, 4])
        frame = runs("goal", data_root=tmp)
        with pytest.raises(KeyError):
            across_replicates(frame, keys=["not_a_column"])


def test_across_replicates_reports_n():
    """n is how many worlds went into a band; n=1 gives lo == hi == mean."""
    with tempfile.TemporaryDirectory() as tmp:
        root = pathlib.Path(tmp) / "goal"
        _write_sweep(root, "demo", [3, 4], offset=0.0)
        _write_sweep(root, "demo", [3, 4], offset=1.0)
        points = across_replicates(runs("goal", data_root=tmp), keys=["n_genes"])
        assert points["n"].tolist() == [2, 2]
        row = points[points.n_genes == 3].iloc[0]
        assert row["full_array_entropy lo"] < row["full_array_entropy hi"]


def test_minmax_is_refused_with_a_reason():
    """min/max was removed: its expected width grows with n, so bands backed by
    different numbers of worlds were not comparable."""
    with tempfile.TemporaryDirectory() as tmp:
        _write_sweep(pathlib.Path(tmp) / "goal", "demo", [3, 4])
        frame = runs("goal", data_root=tmp)
        with pytest.raises(ValueError, match="min/max was removed"):
            across_replicates(frame, keys=["n_genes"], spread="minmax")


def test_one_world_gives_no_band():
    """A single world supports no band; NaN is the honest answer, not zero."""
    with tempfile.TemporaryDirectory() as tmp:
        _write_sweep(pathlib.Path(tmp) / "goal", "demo", [3, 4])
        points = across_replicates(runs("goal", data_root=tmp), keys=["n_genes"])
        assert (points["n"] == 1).all()
        assert points["full_array_entropy lo"].isna().all()


def test_tags_become_columns():
    """Tags record what a run MEANT, which the config cannot express.

    A sampled gene-set strategy draws the number of genes per cell, so the
    realised mean drifts from the requested one and cannot be rounded back.
    Without the target recorded, every world lands on its own x and no point
    has more than one run behind it.
    """
    with tempfile.TemporaryDirectory() as tmp:
        root = pathlib.Path(tmp) / "goal"
        configs = sweep(dict(RECEPTOR), n_genes=[3, 4],
                        tags=[{"arm": "a", "target_genes": 1.5},
                              {"arm": "b", "target_genes": 2.5}])
        logger = SweepLogger(configs, "demo", root)
        for config in configs:
            logger.run_logger(config).save_results(RESULT)
        frame = runs("goal", data_root=tmp)
        assert frame["arm"].tolist() == ["a", "b"]
        assert frame["target_genes"].tolist() == [1.5, 2.5]
        assert across_replicates(frame, keys=["arm"]).shape[0] == 2


def test_tags_never_reach_the_model():
    """Tags are descriptive. Two configs differing only in tags are the same run."""
    a = Config(**dict(RECEPTOR, tags={"note": "left"}))
    b = Config(**dict(RECEPTOR, tags={"note": "right"}))
    assert a.receptor_indices == b.receptor_indices
    physics = {k: v for k, v in a.to_dict().items() if k != "tags"}
    assert physics == {k: v for k, v in b.to_dict().items() if k != "tags"}
