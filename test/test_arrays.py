"""Receptor array composition: combinatorics that a figure's x-axis depends on.

The pool of distinct receptors saturates fast at small n_genes. Asking for more
than exists used to return a SHORT array under the requested label, which put a
run on the wrong curve. These lock that down.
"""
import pytest
import torch

from model.arrays import build_heteromer_array, pool_size


@pytest.mark.parametrize("n_genes", range(1, 9))
@pytest.mark.parametrize("k_sub", [2, 3, 5])
@pytest.mark.parametrize("interface", [False, True])
def test_pool_size_matches_the_generator(n_genes, k_sub, interface):
    """The closed form must agree with enumeration, or sweeps mis-size."""
    expected = pool_size(n_genes, k_sub, interface)
    built = build_heteromer_array(n_genes, k_sub, expected, "cascading", 0, interface)
    assert built.shape[0] == expected
    assert len({tuple(r) for r in built.tolist()}) == expected


def test_asking_past_the_pool_raises():
    """2 genes, k_sub=5, interface: only 8 rings exist. R=10 must not quietly give 8."""
    assert pool_size(2, 5, True) == 8
    with pytest.raises(ValueError, match="exceeds the pool"):
        build_heteromer_array(2, 5, 10, "cascading", 0, True)


@pytest.mark.parametrize("n_genes,k_sub,interface",
                         [(5, 5, True), (5, 5, False), (7, 5, True), (3, 5, False)])
def test_ratio_one_is_the_homomer_array(n_genes, k_sub, interface):
    """R == n_genes under cascading IS the homomer condition.

    Cascading fills complexity tiers in order and tier 1 holds exactly n_genes
    rings, so no separate homomer sweep is needed. This is why fig1's ratio-1
    curve doubles as its homomer reference.
    """
    cascading = build_heteromer_array(n_genes, k_sub, n_genes, "cascading", 0, interface)
    homomers = torch.tensor([[g] * k_sub for g in range(n_genes)], dtype=torch.long)
    assert sorted(map(tuple, cascading.tolist())) == sorted(map(tuple, homomers.tolist()))
