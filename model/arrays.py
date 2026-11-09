"""Which heteromer combinations populate the receptor array.

A receptor is k_sub subunits drawn from n_genes gene products. Two conventions:

  multiset model   subunit order does not matter, a receptor is a sorted tuple
  interface model  subunits sit on a ring; two arrangements are the same
                   receptor iff one is a cyclic rotation of the other.
                   Reflections stay DISTINCT because the +/- face asymmetry
                   breaks mirror symmetry.

Complexity tier u = number of distinct genes in a receptor (u=1 homomer, u=2
dimer, and so on). "cascading" fills the array ascending in u, which is what
makes a small array resemble a biological repertoire rather than a random one.
"""
import itertools
import random
from math import comb, gcd

import torch


def _canonical_rotation(seq):
    """Lexicographically minimum rotation: the ring's canonical representative."""
    return min(seq[i:] + seq[:i] for i in range(len(seq)))


def _positive_compositions(total, parts):
    """Ordered compositions of `total` into exactly `parts` positive integers."""
    for dividers in itertools.combinations(range(1, total), parts - 1):
        prev, comp = 0, []
        for d in dividers:
            comp.append(d - prev)
            prev = d
        yield tuple(comp + [total - prev])


def _tier_canonical_forms(n_genes, k_sub, n_unique):
    """Canonical ring arrangements using EXACTLY n_unique distinct genes.

    One complexity tier without touching the others, so cascading can stop
    early instead of enumerating the whole pool (the high tiers dominate it).
    """
    forms = set()
    for genes in itertools.combinations(range(n_genes), n_unique):
        for counts in _positive_compositions(k_sub, n_unique):
            multiset = tuple(g for g, c in zip(genes, counts) for _ in range(c))
            for perm in set(itertools.permutations(multiset)):
                forms.add(_canonical_rotation(perm))
    return forms


def ordered_random(n_genes, k_sub, n_sensors, seed=None):
    """Uniform sample of distinct ring arrangements."""
    canonical = set()
    for combo in itertools.combinations_with_replacement(range(n_genes), k_sub):
        for perm in set(itertools.permutations(combo)):
            canonical.add(_canonical_rotation(perm))
    pool = sorted(canonical)
    rng = random.Random(seed)
    selected = pool if n_sensors > len(pool) else rng.sample(pool, n_sensors)
    return torch.tensor(selected, dtype=torch.long)


def ordered_cascading(n_genes, k_sub, n_sensors, seed=None):
    """Ring arrangements, quota filled ascending by complexity tier."""
    rng = random.Random(seed)
    selected, remaining = [], n_sensors
    for n_unique in range(1, k_sub + 1):
        if remaining <= 0:
            break
        # sorted() makes the seeded shuffle reproducible across platforms,
        # which set-insertion order would not be.
        available = sorted(_tier_canonical_forms(n_genes, k_sub, n_unique))
        rng.shuffle(available)
        selected.extend(available[:remaining])
        remaining -= min(remaining, len(available))
    return torch.tensor(selected, dtype=torch.long)


def multiset_cascading(n_genes, k_sub, n_sensors, seed=None):
    """Sorted-tuple receptors, quota filled ascending by complexity tier."""
    buckets = {}
    for combo in itertools.combinations_with_replacement(range(n_genes), k_sub):
        buckets.setdefault(len(set(combo)), []).append(combo)

    rng = random.Random(seed)
    selected, remaining = [], n_sensors
    for n_unique in sorted(buckets):
        if remaining <= 0:
            break
        available = buckets[n_unique]
        rng.shuffle(available)
        selected.extend(available[:remaining])
        remaining -= min(remaining, len(available))
    return torch.tensor(selected, dtype=torch.long)


def multiset_random(n_genes, k_sub, n_sensors, seed=None):
    """Uniform sample of sorted-tuple receptors, by reservoir over the pool."""
    rng = random.Random(seed)
    reservoir = []
    for i, combo in enumerate(
            itertools.combinations_with_replacement(range(n_genes), k_sub)):
        if i < n_sensors:
            reservoir.append(combo)
        else:
            j = rng.randint(0, i)
            if j < n_sensors:
                reservoir[j] = combo
    return torch.tensor(reservoir, dtype=torch.long)


_STRATEGIES = {
    (False, "uniform_random"): multiset_random,
    (False, "cascading"):      multiset_cascading,
    (True,  "uniform_random"): ordered_random,
    (True,  "cascading"):      ordered_cascading,
}


def pool_size(n_genes, k_sub, use_interface_model=False):
    """How many DISTINCT receptors exist for this gene count and model.

    multiset model   combinations with replacement: C(n + k - 1, k).
    interface model  necklaces of length k over n symbols, rotations identified
                     and reflections kept distinct, so by Burnside
                     (1/k) * sum_{d | k} phi(d) * n^(k/d).

    The pool saturates fast at small n_genes: 2 genes with k_sub=5 gives only 8
    distinct rings, so a sweep over R = 2, 4, 6, 8, 10 cannot reach its last
    point. Check this before asking for an array.
    """
    if not use_interface_model:
        return comb(n_genes + k_sub - 1, k_sub)
    total = 0
    for d in range(1, k_sub + 1):
        if k_sub % d == 0:
            # Euler totient of d, by definition rather than by import.
            phi = sum(1 for i in range(1, d + 1) if gcd(i, d) == 1)
            total += phi * n_genes ** (k_sub // d)
    return total // k_sub


def build_heteromer_array(n_genes, k_sub, R_target, strategy="cascading",
                          seed=None, use_interface_model=False):
    """(R_target, k_sub) long tensor of subunit indices.

    Raises when the pool is smaller than R_target. Returning a short array
    instead would silently relabel the point: a sweep axis would read R=100
    for an array of 36, and the figure would be wrong rather than missing.
    """
    array = _STRATEGIES[(bool(use_interface_model), strategy)](
        n_genes, k_sub, R_target, seed=seed)
    if array.shape[0] < R_target:
        raise ValueError(
            f"R_target={R_target} exceeds the pool of {array.shape[0]} distinct "
            f"receptors for n_genes={n_genes}, k_sub={k_sub}, "
            f"interface={bool(use_interface_model)}.")
    return array
