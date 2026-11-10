# Documented in:
#   doc/theory/09_cell_arrays.md            (REFERENCE for this file: sampling, choices)
#   doc/theory/07_optimization_pipeline.md  (stage 3b: where it sits in the pipeline)
#   doc/theory/02_biophysics_mwc.md         (receptor activation this layer pools over)
"""
cells.py — Cell-level readout layered on top of the receptor array.

Point of view shift: the sensory unit is no longer a receptor but a CELL, defined
by the set of subunit genes G_c it expresses.  A cell assembles EVERY receptor its
genes allow, so the array is specified by C gene sets rather than by R receptors.

Three objects:
  CellArray    — gene sets → deduplicated receptor pool + abundance matrix W (C, R_pool)
  CellReadout  — (B, R_pool) receptor activity → (B, C) cell activity
  cell_activity — chunked physics+readout so peak memory is O(B·L·chunk), not O(B·L·R_pool)

Terminology (see doc/theory/01_nomenclature.md):
  k_sub    number of subunits per receptor (5)
  G_c      set of genes expressed by cell c;  g = |G_c|
  R_pool   size of the deduplicated union of all cells' repertoires
  w_cr     abundance of receptor r in cell c, normalised so sum_r w_cr = 1
  S_bc     drive of cell c on sniff b: the abundance-weighted open fraction
  theta    firing threshold, ONE scalar shared by all cells (learnable)
  T_cell   sharpness of the cell firing step (annealed, as for the receptor step)

Repertoire size grows fast in g: a cell expressing g genes makes C(g+k_sub-1, k_sub)
multiset receptors (g=2 -> 6, g=5 -> 126, g=8 -> 792), or (1/k_sub) sum_{d|k_sub}
phi(d) g^(k_sub/d) ring arrangements under the interface model.  R_pool is the
binding cost of the whole pipeline; see doc/theory/06_computational_limits.md.
"""
import itertools
import math
import random
from math import factorial
from typing import List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from model.arrays import _canonical_rotation


# ---------------------------------------------------------------------------
# Repertoire expansion
# ---------------------------------------------------------------------------

def expand_gene_set(genes: Sequence[int], k_sub: int,
                    use_interface_model: bool = False) -> List[Tuple[int, ...]]:
    """Every receptor a cell expressing `genes` can assemble.

    Standard model: unordered multisets of size k_sub drawn from `genes`.
        {0,2}, k_sub=5 -> (0,0,0,0,0) (0,0,0,0,2) (0,0,0,2,2)
                          (0,0,2,2,2) (0,2,2,2,2) (2,2,2,2,2)

    Interface model: canonical cyclic ring arrangements (rotations identified,
    reflections distinct — the +/- face asymmetry breaks mirror symmetry, same
    convention as geometry.generate_ordered_receptor_indices).

    Returns a sorted list, so the pool order is deterministic across runs.
    """
    genes = sorted(set(int(g) for g in genes))
    if not genes:
        raise ValueError("expand_gene_set: a cell must express at least one gene.")

    if not use_interface_model:
        return sorted(itertools.combinations_with_replacement(genes, k_sub))

    forms = {_canonical_rotation(w) for w in itertools.product(genes, repeat=k_sub)}
    return sorted(forms)


def _multiset_multiplicity(receptor: Tuple[int, ...], k_sub: int) -> int:
    """Number of ordered arrangements collapsing onto this multiset: k! / prod(n_i!)."""
    counts: dict = {}
    for u in receptor:
        counts[u] = counts.get(u, 0) + 1
    denom = 1
    for n in counts.values():
        denom *= factorial(n)
    return factorial(k_sub) // denom


def _ring_multiplicity(receptor: Tuple[int, ...]) -> int:
    """Number of distinct rotations of a canonical ring word (= k_sub / |stabiliser|)."""
    k = len(receptor)
    return len({receptor[i:] + receptor[:i] for i in range(k)})


def repertoire_weights(receptors: Sequence[Tuple[int, ...]], n_genes_expressed: int,
                       k_sub: int, stoichiometry: str = "multinomial",
                       use_interface_model: bool = False) -> List[float]:
    """Abundance of each receptor in a cell, normalised to sum to 1.

    stoichiometry='multinomial' (random assembly): all subunits are produced in
      equal amounts and assemble independently into the k_sub slots, so each of
      the g^k_sub ordered words is equally likely and a receptor's abundance is
      proportional to how many words collapse onto it:
        multiset model  w_r ∝ k_sub! / prod_i n_i!     (n_i = copies of gene i)
        interface model w_r ∝ number of distinct rotations of the ring word
      This suppresses homomers (1 word out of k_sub!) relative to mixed receptors.

    stoichiometry='uniform': every receptor type the cell can make is present in
      equal amount; implies per-type assembly control rather than random mixing.
    """
    if stoichiometry == "uniform":
        w = 1.0 / len(receptors)
        return [w] * len(receptors)
    if stoichiometry != "multinomial":
        raise ValueError(f"Unknown stoichiometry {stoichiometry!r}. "
                         f"Choose 'multinomial' or 'uniform'.")

    if use_interface_model:
        mult = [float(_ring_multiplicity(r)) for r in receptors]
    else:
        mult = [float(_multiset_multiplicity(r, k_sub)) for r in receptors]
    total = sum(mult)
    return [m / total for m in mult]


# ---------------------------------------------------------------------------
# Cell array
# ---------------------------------------------------------------------------

class CellArray:
    """A set of cells, expanded into a shared receptor pool plus abundances.

    Attributes:
        gene_sets:        list of C sorted gene tuples.
        receptor_indices: (R_pool, k_sub) long tensor — the DEDUPLICATED union of
                          every cell's repertoire.  Fed unchanged to the existing
                          physics; a receptor shared by several cells is simulated once.
        W:                (C, R_pool) float tensor of abundances, rows summing to 1.
                          W[c, r] = 0 when cell c cannot assemble receptor r.
    """

    def __init__(self, gene_sets: Optional[Sequence[Sequence[int]]], k_sub: int,
                 stoichiometry: str = "multinomial",
                 use_interface_model: bool = False,
                 repertoires: Optional[Sequence[Sequence[Sequence[int]]]] = None):
        """Build from GENE SETS (the usual way) or from explicit REPERTOIRES.

        gene_sets: one gene set per cell; each is expanded into every receptor those
          genes allow (expand_gene_set) and weighted by `stoichiometry`. This is the
          biological model: you choose what the cell EXPRESSES, the repertoire follows.

        repertoires: one explicit receptor list per cell, bypassing the expansion.
          Weights are UNIFORM over the receptors you listed — you have specified the
          repertoire directly, so `stoichiometry` does not apply and is ignored.
          `gene_sets` is then derived from the receptors, for reporting only.

          This exists because the gene-set model CANNOT express "a cell containing
          exactly one heteromer": expressing the genes of [0,0,0,1,1] necessarily also
          produces every other combination of genes 0 and 1. One gene per cell does give
          exactly one receptor, but only ever the homomer. Explicit repertoires are the
          way to pin an arbitrary receptor into a cell — which is what the cell-vs-
          receptor equivalence check needs (tasks/cells/equivalence).
        """
        self.k_sub = k_sub
        self.stoichiometry = stoichiometry
        self.use_interface_model = use_interface_model
        self.explicit = repertoires is not None

        if self.explicit:
            if gene_sets is not None:
                raise ValueError("CellArray: pass gene_sets OR repertoires, not both.")
            per_cell = []
            for reps in repertoires:
                if not len(reps):
                    raise ValueError("CellArray: a cell must contain at least one receptor.")
                cell = []
                for r in reps:
                    r = tuple(int(u) for u in r)
                    if len(r) != k_sub:
                        raise ValueError(f"receptor {r} has {len(r)} subunits, expected k_sub={k_sub}.")
                    # canonicalise so an explicitly-listed receptor is the SAME object the
                    # gene-set path would have produced (sorted multiset / min rotation).
                    cell.append(_canonical_rotation(r) if use_interface_model else tuple(sorted(r)))
                per_cell.append(sorted(set(cell)))
            # derived, for __repr__ and for anything that inspects which genes are in play
            self.gene_sets = [tuple(sorted({u for r in reps for u in r})) for reps in per_cell]
        else:
            self.gene_sets = [tuple(sorted(set(int(g) for g in gs))) for gs in gene_sets]
            per_cell = [expand_gene_set(gs, k_sub, use_interface_model) for gs in self.gene_sets]

        pool = sorted({r for reps in per_cell for r in reps})
        pos = {r: i for i, r in enumerate(pool)}

        W = torch.zeros(len(self.gene_sets), len(pool), dtype=torch.float32)
        for c, (gs, reps) in enumerate(zip(self.gene_sets, per_cell)):
            weights = ([1.0 / len(reps)] * len(reps) if self.explicit else
                       repertoire_weights(reps, len(gs), k_sub,
                                          stoichiometry, use_interface_model))
            for r, w in zip(reps, weights):
                W[c, pos[r]] = w

        self.receptor_indices = torch.tensor(pool, dtype=torch.long)
        self.W = W

    @property
    def n_cells(self) -> int:
        return self.W.shape[0]

    @property
    def pool_size(self) -> int:
        return self.receptor_indices.shape[0]

    def to(self, device) -> "CellArray":
        self.receptor_indices = self.receptor_indices.to(device)
        self.W = self.W.to(device)
        return self

    def __repr__(self) -> str:
        sizes = [len(gs) for gs in self.gene_sets]
        how = ("repertoires=explicit" if self.explicit
               else f"stoichiometry={self.stoichiometry!r}")
        return (f"CellArray(C={self.n_cells}, R_pool={self.pool_size}, "
                f"genes/cell={min(sizes)}-{max(sizes)}, {how})")


# ---------------------------------------------------------------------------
# Gene-set samplers
# ---------------------------------------------------------------------------

def sample_gene_sets_bernoulli(n_cells: int, n_genes: int,
                               gene_probs: Sequence[float],
                               max_genes: Optional[int] = None,
                               seed: Optional[int] = None) -> List[Tuple[int, ...]]:
    """Each gene is expressed independently with probability gene_probs[u].

    Cells expressing nothing are rejected and redrawn.  max_genes (optional) caps
    the repertoire size per cell — C(g+k_sub-1, k_sub) grows fast in g and R_pool
    is the memory bottleneck.
    """
    rng = random.Random(seed)
    if len(gene_probs) != n_genes:
        raise ValueError(f"gene_probs has length {len(gene_probs)}, expected n_genes={n_genes}.")

    cells: List[Tuple[int, ...]] = []
    while len(cells) < n_cells:
        genes = tuple(u for u in range(n_genes) if rng.random() < gene_probs[u])
        if not genes:
            continue
        if max_genes is not None and len(genes) > max_genes:
            continue
        cells.append(genes)
    return cells


def sample_gene_sets_by_size(n_cells: int, n_genes: int,
                             size_pmf: Sequence[float],
                             gene_probs: Optional[Sequence[float]] = None,
                             seed: Optional[int] = None) -> List[Tuple[int, ...]]:
    """Two-stage sampler: draw the NUMBER of expressed genes, then which ones.

    size_pmf[i] is the (unnormalised) probability that a cell expresses i+1 genes,
    so len(size_pmf) sets the maximum.  This is the hook for an arbitrarily complex
    expression model — a distribution conditioned on how many genes are already
    expressed is exactly a choice of size_pmf.

    gene_probs weights WHICH genes are drawn (successive weighted sampling without
    replacement); None means uniform.  The two laws are independent, so the size
    distribution and the gene-identity distribution can be varied separately.

    Cells are drawn independently: no cell knows what any other cell expressed, and
    a gene can end up expressed nowhere in the array.  Contrast the
    "every gene expressed at least once" designs, which require exactly the kind of
    coordination between cells that a developing tissue has no mechanism for.
    """
    rng = random.Random(seed)
    total = float(sum(size_pmf))
    if total <= 0:
        raise ValueError("size_pmf must have positive total mass.")
    probs = [p / total for p in size_pmf]
    sizes = list(range(1, len(probs) + 1))
    if sizes[-1] > n_genes:
        raise ValueError(f"size_pmf allows {sizes[-1]} genes but only {n_genes} exist.")
    if gene_probs is not None:
        if len(gene_probs) != n_genes:
            raise ValueError(f"gene_probs has length {len(gene_probs)}, expected {n_genes}.")
        if min(gene_probs) < 0 or sum(gene_probs) <= 0:
            raise ValueError("gene_probs must be non-negative with positive total.")

    cells = []
    for _ in range(n_cells):
        g = rng.choices(sizes, weights=probs, k=1)[0]
        if gene_probs is None:
            cells.append(tuple(sorted(rng.sample(range(n_genes), g))))
            continue
        pool, weights, chosen = list(range(n_genes)), list(gene_probs), []
        for _ in range(g):
            i = rng.choices(range(len(pool)), weights=weights, k=1)[0]
            chosen.append(pool.pop(i))
            weights.pop(i)
        cells.append(tuple(sorted(chosen)))
    return cells


def size_pmf_for_mean(n_genes: int, mean: float, family: str = "uniform",
                      tol: float = 1e-10) -> List[float]:
    """Distribution over the NUMBER of genes a cell expresses, with a target mean.

    The mean is the experimental knob: it moves the array continuously from "one
    gene per cell" to "everything everywhere", so curves from different laws can be
    compared at matched expression level.

    family='uniform'     flat over {1..M}.  Only means (M+1)/2 are reachable, i.e.
                         1, 1.5, ..., (n_genes+1)/2.
    family='exponential' p_i proportional to exp(lambda*i) over {1..n_genes}, with
                         lambda solved for the requested mean.  lambda<0 favours
                         small cells, lambda=0 is flat over the full range.  Any
                         mean strictly inside (1, n_genes) is reachable.
    """
    if not 1 <= mean <= n_genes:
        raise ValueError(f"mean must lie in [1, {n_genes}], got {mean}.")
    sizes = list(range(1, n_genes + 1))

    if family == "uniform":
        m = 2 * mean - 1
        if abs(m - round(m)) > 1e-9 or not 1 <= round(m) <= n_genes:
            reachable = [(k + 1) / 2 for k in range(1, n_genes + 1)]
            raise ValueError(f"family='uniform' cannot reach mean {mean}. "
                             f"Reachable means: {reachable}.")
        m = round(m)
        return [1.0 / m if i <= m else 0.0 for i in sizes]

    if family != "exponential":
        raise ValueError(f"Unknown size family {family!r}. Use 'uniform' or 'exponential'.")

    def weights_at(lam: float) -> List[float]:
        logits = [lam * i for i in sizes]
        shift = max(logits)                 # subtract the max: exp never overflows
        return [math.exp(l - shift) for l in logits]

    def mean_at(lam: float) -> float:
        w = weights_at(lam)
        return sum(i * x for i, x in zip(sizes, w)) / sum(w)

    if abs(mean - 1) < tol:
        return [1.0] + [0.0] * (n_genes - 1)
    if abs(mean - n_genes) < tol:
        return [0.0] * (n_genes - 1) + [1.0]
    lo, hi = -1e3, 1e3                       # mean_at is increasing in lambda
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if mean_at(mid) < mean:
            lo = mid
        else:
            hi = mid
    w = weights_at(0.5 * (lo + hi))
    total = sum(w)
    return [x / total for x in w]


def gene_expression_probs(n_genes: int, family: str = "uniform",
                          ratio: float = 1.0) -> List[float]:
    """How likely each GENE is to be picked, independent of how many are picked.

    family='uniform'     every gene equally likely.
    family='exponential' geometric decay across the gene index, with
                         `ratio` = P(first gene) / P(last gene).  ratio=1 is uniform.

    A skewed law makes some subunits common and others rare, so the array explores
    a biased corner of receptor space rather than all of it evenly.
    """
    if n_genes < 1:
        raise ValueError("n_genes must be positive.")
    if family == "uniform" or n_genes == 1:
        return [1.0 / n_genes] * n_genes
    if family != "exponential":
        raise ValueError(f"Unknown gene family {family!r}. Use 'uniform' or 'exponential'.")
    if ratio <= 0:
        raise ValueError("ratio must be positive.")
    w = [ratio ** (-u / (n_genes - 1)) for u in range(n_genes)]
    total = sum(w)
    return [x / total for x in w]


def build_cell_array(
    k_sub: int,
    n_genes: int,
    gene_sets: Optional[Sequence[Sequence[int]]] = None,
    n_cells: Optional[int] = None,
    strategy: str = "bernoulli",
    gene_probs: Optional[Sequence[float]] = None,
    size_pmf: Optional[Sequence[float]] = None,
    max_genes: Optional[int] = None,
    seed: Optional[int] = None,
    stoichiometry: str = "multinomial",
    use_interface_model: bool = False,
    repertoires: Optional[Sequence[Sequence[Sequence[int]]]] = None,
) -> CellArray:
    """Unified entry point: explicit repertoires, explicit gene sets, or sampled ones.

    repertoires wins if given (the receptor list per cell is stated outright);
    otherwise gene_sets; otherwise n_cells gene sets are sampled with `strategy`.
    strategy='bernoulli' uses gene_probs (defaults to uniform 2/n_genes, i.e. two
    expressed genes per cell on average); strategy='size_pmf' uses size_pmf, and
    gene_probs then weights WHICH genes are drawn at the chosen size.
    """
    if repertoires is not None:
        return CellArray(None, k_sub, use_interface_model=use_interface_model,
                         repertoires=repertoires)

    if gene_sets is None:
        if n_cells is None:
            raise ValueError("build_cell_array: provide gene_sets or n_cells.")
        if strategy == "bernoulli":
            if gene_probs is None:
                gene_probs = [2.0 / n_genes] * n_genes
            gene_sets = sample_gene_sets_bernoulli(n_cells, n_genes, gene_probs,
                                                   max_genes=max_genes, seed=seed)
        elif strategy == "size_pmf":
            if size_pmf is None:
                raise ValueError("strategy='size_pmf' requires size_pmf.")
            gene_sets = sample_gene_sets_by_size(n_cells, n_genes, size_pmf,
                                                 gene_probs=gene_probs, seed=seed)
        else:
            raise ValueError(f"Unknown strategy {strategy!r}. Choose 'bernoulli' or 'size_pmf'.")

    return CellArray(gene_sets, k_sub, stoichiometry=stoichiometry,
                     use_interface_model=use_interface_model)


# ---------------------------------------------------------------------------
# Readout
# ---------------------------------------------------------------------------

class CellReadout(nn.Module):
    """Receptor activity (B, R_pool) -> cell activity (B, C).

        A_bc = sum_r w_cr p_br        abundance-weighted open fraction

    w_cr is how much of receptor type r cell c carries, so A_bc is the
    probability that a randomly chosen open receptor in cell c is of type r,
    summed: a stochastic cell whose firing probability IS its weighted open
    fraction. The MI estimators subtract its response noise, which an
    entropy-only objective would instead reward. With one receptor type per
    cell this reproduces the receptor model exactly.

    The reduction over the pool is LINEAR in the per-receptor contribution, so
    the pool can be walked in chunks and the accumulator summed. That is what
    bounds memory (see cell_activity).
    """

    def __init__(self, W: torch.Tensor, k_sub: int = 5):
        super().__init__()
        self.k_sub = k_sub
        # Buffer, not parameter: it moves with .to(device) and is checkpointed,
        # but carries no gradient. The repertoire is fixed, never learned.
        self.register_buffer("W", W)

    @property
    def n_cells(self) -> int:
        return self.W.shape[0]

    def accumulate(self, p_chunk: torch.Tensor, r0: int, r1: int) -> torch.Tensor:
        """Contribution of receptor-pool slice [r0:r1] to the (B, C) accumulator."""
        return p_chunk @ self.W[:, r0:r1].T

    def finalize(self, acc: torch.Tensor) -> torch.Tensor:
        return acc.clamp(0.0, 1.0)

    def drive(self, activity: torch.Tensor) -> torch.Tensor:
        return self.accumulate(activity, 0, self.W.shape[1])

    def forward(self, activity: torch.Tensor) -> torch.Tensor:
        return self.finalize(self.drive(activity))


# ---------------------------------------------------------------------------
# Chunked physics + readout
# ---------------------------------------------------------------------------

def cell_activity(physics, readout: CellReadout,
                  energies: torch.Tensor, concentrations: torch.Tensor,
                  receptor_indices: torch.Tensor,
                  pre_gathered: bool = False,
                  chunk_size: Optional[int] = None,
                  recompute: bool = False,
                  composition: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Receptor physics + cell pooling, traversing the receptor pool in chunks.

    The physics tensors are (B, L, R_pool[, k_sub]) — R_pool is the union of every
    cell's repertoire and is the memory bottleneck of the cell picture.  Because
    CellReadout.accumulate is linear in the pool axis, the pool can be walked in
    slices of `chunk_size` receptors and the (B, C) accumulator summed, so the
    peak forward tensor is O(B·L·chunk) instead of O(B·L·R_pool).

    chunk_size=None runs the pool in one pass (identical result, matching the
    receptor-array behaviour).

    recompute=True gradient-checkpoints each chunk so its activations are
    recomputed in backward rather than retained: without it the retained graph is
    still O(B·L·R_pool) during training and chunking only helps under no_grad.
    Exact — no effect on values or gradients (same convention as
    Config.recompute_backward).
    """
    R_pool = receptor_indices.shape[0]
    if chunk_size is None or chunk_size >= R_pool:
        return readout(physics(energies, concentrations, receptor_indices,
                               pre_gathered=pre_gathered, composition=composition))

    def _chunk(r0: int, r1: int) -> torch.Tensor:
        if composition is not None:
            # Slice the composition COLUMNS (one per receptor); the per-source energy
            # tensor is shared by every chunk and must not be sliced.
            p = physics(energies, concentrations, receptor_indices[r0:r1],
                        composition=composition[:, r0:r1])
        else:
            e = energies[:, :, r0:r1] if pre_gathered else energies
            p = physics(e, concentrations, receptor_indices[r0:r1], pre_gathered=pre_gathered)
        return readout.accumulate(p, r0, r1)

    acc = None
    do_ckpt = recompute and torch.is_grad_enabled() and energies.requires_grad
    for r0 in range(0, R_pool, chunk_size):
        r1 = min(r0 + chunk_size, R_pool)
        if do_ckpt:
            # checkpoint needs tensor args to track; r0/r1 ride along as ints.
            part = checkpoint(lambda a, b: _chunk(int(a), int(b)),
                              torch.tensor(r0), torch.tensor(r1), use_reentrant=False)
        else:
            part = _chunk(r0, r1)
        acc = part if acc is None else acc + part

    return readout.finalize(acc)


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------





