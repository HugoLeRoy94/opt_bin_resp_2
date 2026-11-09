"""Which genes land in which cell, and what each cell may build from them.

The ONLY shared code under tasks/. Every parameter of every experiment is
written out in the script that runs it; nothing is hidden here except the
combinatorics of assigning genes to cells.

Two ways to draw gene sets, and they ask different questions:

  complete coverage  every gene is guaranteed to appear in at least one cell,
                     via a mandatory anchor dealt round-robin. The repertoire is
                     coordinated across the population.
  bernoulli          each gene is expressed independently at probability p, so
                     the number per cell is INDUCED rather than chosen, and a
                     gene can end up expressed nowhere.

And two ways a cell turns its genes into receptors:

  heteromers  the gene-set path. A cell holding {u, v} builds every receptor
              those genes allow, mixed subunits included.
  homomers    homomers_only() states the receptor list outright. The gene-set
              path cannot express this, because expressing u and v necessarily
              also builds every mixed receptor they allow.

Everything here is pure: same arguments, same output, no global state.
"""
import math
from collections import Counter

import torch

def complete_coverage_gene_sets(n_genes, n_cells, genes_per_cell, seed):
    """Random gene sets, but every gene is guaranteed to appear in at least one cell.

    Each cell gets one mandatory ANCHOR gene, dealt round-robin over the genes so
    multiplicities are near-balanced, then fills the rest by random priority. This
    needs cells to know what other cells expressed, which a developing tissue cannot
    do — it is a controlled baseline, not a biological model. Use random_gene_sets
    for the independent version.

    Reusing one seed across genes_per_cell makes the sets NESTED: the g=2 array is
    the g=1 array with one gene added per cell. That is what makes an expression
    sweep a sweep rather than five unrelated arrays.

    Returns a tuple of n_cells sorted gene tuples.
    """
    if not 1 <= genes_per_cell <= n_genes:
        raise ValueError(f"Need 1 <= genes_per_cell <= n_genes, got {genes_per_cell}/{n_genes}.")
    if n_cells < n_genes:
        raise ValueError(f"Complete coverage needs n_cells >= n_genes, got {n_cells} < {n_genes}.")
    generator = torch.Generator().manual_seed(seed)
    priority = torch.rand(n_cells, n_genes, generator=generator)
    permutation = torch.randperm(n_genes, generator=generator)
    anchors = permutation[torch.arange(n_cells) % n_genes]
    anchors = anchors[torch.randperm(n_cells, generator=generator)]
    priority.scatter_(1, anchors[:, None], -1.)
    genes = priority.topk(genes_per_cell, dim=1, largest=False).indices.sort(dim=1).values
    return tuple(map(tuple, genes.tolist()))


def complete_coverage_gene_sets_mixed(n_genes, n_cells, mean, seed):
    """Complete coverage at a NON-INTEGER mean genes per cell.

    complete_coverage_gene_sets gives every cell exactly `genes_per_cell` genes, so
    the mean is forced to an integer and the covering arm cannot share an x axis with
    a continuously-parameterised one. Here `n_big = round(frac * n_cells)` cells get
    floor(mean)+1 genes and the rest get floor(mean).

    Coverage survives because the anchor mechanism is untouched: every cell still
    gets one mandatory gene dealt round-robin, so with n_cells >= n_genes every gene
    is expressed somewhere whatever the sizes are. Which cells get the extra gene is
    itself shuffled, so the big cells are not systematically the ones holding any
    particular anchor.

    Returns (gene_sets, realised_mean). The realised mean is exact when frac*n_cells
    is a whole number and rounds otherwise, so plot the realised value, not the
    target. At integer `mean` this reduces exactly to complete_coverage_gene_sets.
    """
    if not 1 <= mean <= n_genes:
        raise ValueError(f"mean must lie in [1, {n_genes}], got {mean}.")
    if n_cells < n_genes:
        raise ValueError(f"Complete coverage needs n_cells >= n_genes, got {n_cells} < {n_genes}.")
    base = int(math.floor(mean + 1e-9))
    n_big = int(round((mean - base) * n_cells))
    if n_big == 0:
        return complete_coverage_gene_sets(n_genes, n_cells, base, seed), float(base)
    if base + 1 > n_genes:
        raise ValueError(f"mean {mean} needs {base + 1} genes but only {n_genes} exist.")

    generator = torch.Generator().manual_seed(seed)
    priority = torch.rand(n_cells, n_genes, generator=generator)
    permutation = torch.randperm(n_genes, generator=generator)
    anchors = permutation[torch.arange(n_cells) % n_genes]
    anchors = anchors[torch.randperm(n_cells, generator=generator)]
    priority.scatter_(1, anchors[:, None], -1.)

    sizes = torch.full((n_cells,), base, dtype=torch.long)
    sizes[torch.randperm(n_cells, generator=generator)[:n_big]] = base + 1
    gene_sets = tuple(
        tuple(sorted(priority[c].topk(int(sizes[c]), largest=False).indices.tolist()))
        for c in range(n_cells))
    return gene_sets, sum(map(len, gene_sets)) / n_cells


def homomers_only(gene_sets, k_sub):
    """Turn gene sets into explicit receptor lists containing only homomers.

    A cell expressing {u, v} gets [(u,)*k_sub, (v,)*k_sub] and nothing else. Pass the
    result as RunConfig(cell_receptors=...), which bypasses repertoire expansion and
    weights the listed receptors uniformly.

    The gene-set path CANNOT express this: expressing u and v necessarily also builds
    every mixed receptor they allow.
    """
    return tuple(tuple((gene,) * k_sub for gene in genes) for genes in gene_sets)


def count_alphabet(gene_sets):
    """Size of the grouped count alphabet these cells produce: prod_j (n_j + 1).

    Cells with an IDENTICAL gene set are one group, and the grouped estimator keeps
    only how many of them fired. That loses nothing: the count vector is a sufficient
    statistic of the labelled response for the input, and the label term cancels
    between H(Y) = H(K) + d and H(Y|X) = H(K|X) + d, so grouped MI equals the
    labelled-codeword MI exactly.

    So log2(count_alphabet) is a real ceiling on what the ARRAY can carry, however
    good the receptors are. Thirty cells with one shared repertoire are thirty noisy
    copies of one measurement. The ceiling is not monotonic in genes per cell because
    the number of distinct repertoires available, C(G, g), is not — print it next to
    any grouped-MI curve to tell "saturating its combinatorics" from "limited by the
    receptors".
    """
    return math.prod(n + 1 for n in Counter(gene_sets).values())


def heterogeneous_conc_means(n_points, n_ligands, log10_span, seed):
    """One per-ligand log10 concentration-mean vector per sweep point, in +/- span.

    log10_span = 0 gives the flat channel (every ligand the same typical abundance).
    The environment initialises its EC50 from the MEAN of the vector, so a symmetric
    spread moves which ligands sit near saturation rather than moving the whole
    dynamic range.
    """
    generator = torch.Generator().manual_seed(seed)
    return [tuple(((torch.rand(n_ligands, generator=generator) * 2.0 - 1.0)
                   * log10_span).tolist())
            for _ in range(n_points)]


def bernoulli_p_for_mean(n_genes, mean, p_floor=0.02):
    """Per-gene expression probability giving the requested mean genes per cell.

    Each gene is expressed independently at probability p, so the NUMBER per cell is
    induced (Binomial), never chosen. Cells expressing nothing are redrawn, so the
    realised mean is the ZERO-TRUNCATED one, G*p / (1 - (1-p)^G), not G*p. Setting
    p = mean/G would overshoot badly at small means.

    p_floor exists because mean -> 1 needs p -> 0, and at p -> 0 almost every draw is
    an empty cell that gets rejected, so the sampler stalls. The floor bounds the
    rejection rate (acceptance ~ G*p_floor) at the price of a realised mean a little
    above 1. Plot the REALISED mean, which bernoulli_gene_sets reports.
    """
    if not 1 <= mean <= n_genes:
        raise ValueError(f"mean must lie in [1, {n_genes}], got {mean}.")
    realised = lambda p: n_genes * p / (1 - (1 - p) ** n_genes)
    if mean <= realised(p_floor):
        return p_floor
    lo, hi = p_floor, 1.0 - 1e-12
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if realised(mid) < mean else (lo, mid)
    return 0.5 * (lo + hi)


def bernoulli_gene_sets(n_genes, n_cells, mean, seed, p_floor=0.02):
    """Independent per-gene expression at a uniform probability, size induced.

    The counterpart to complete_coverage_gene_sets: no anchor, no size law, and no
    coordination between cells. A gene can end up expressed nowhere, and the spread
    of cell sizes is whatever the Binomial gives rather than something imposed.

    Returns (gene_sets, p, realised_mean).
    """
    from model.cells import sample_gene_sets_bernoulli
    p = bernoulli_p_for_mean(n_genes, mean, p_floor)
    sets = tuple(sample_gene_sets_bernoulli(n_cells, n_genes, [p] * n_genes, seed=seed))
    return sets, p, sum(map(len, sets)) / len(sets)
