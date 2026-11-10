"""fig3: can the MI maximum be moved? Everything the three k_sub arms share.

fig2 found the maximum sits at 2 genes per cell for every G and every C >= 20,
and will not move. The reason is structural: under the mean readout the
receptor POOL and the averaging WIDTH are the same function of g, the number
of genes a cell expresses. Benefit and cost rise together.

  benefit   more genes across the population means more distinct receptor
            types exist, so the array discriminates more
  cost      a cell's firing probability is the abundance-weighted AVERAGE over
            the receptors it carries, so a cell holding more types has each
            one's contribution diluted toward the mean

k_sub changes how fast each rises, and not by the same amount. At G = 5:

                receptors per CELL        distinct receptors
                (the dilution width)      in the ARRAY (the pool)
    k_sub=2    [1,  3,  6,  10,  15]     [5, 15,  15,  15,  15]
    k_sub=3    [1,  4, 11,  24,  45]     [5, 25,  45,  45,  45]
    k_sub=5    [1,  8, 51, 208, 629]     [5, 65, 365, 605, 629]

Ratio of pool gain to dilution gain, step by step. A step pays off above 1:

    k_sub    1->2   2->3   3->4   4->5
      2      1.00   0.50   0.60   0.67
      3      1.25   0.65   0.46   0.53
      5      1.62   0.88   0.41   0.34

PREDICTION: the peak moves LEFT as k_sub falls, to g=1 at k=2 and g=2 at k=3
and k=5. Lowering k_sub does slow dilution, but it slows the pool by at least
as much, and only the ratio matters.

At k=2 the pool is exhausted at g=2, which makes that arm a PURE-DILUTION
CONTROL: everything past g=2 is dilution with zero pool gain behind it. If MI
still falls there, dilution alone is sufficient to produce the decline.

The three arms share CELL_SEED, so they get IDENTICAL gene sets and k_sub is
the only thing that differs.
"""
import time

import numpy as np

from tasks._lib.budget import counting_budget_report
from tasks._lib.repertoires import (complete_coverage_gene_sets, count_alphabet,
                                    heterogeneous_conc_means)
from train.config import sweep
from train.sweep import Sweep

WORLDS = 1              # raise this for more replicates; nothing else changes

# ─── The design grid ────────────────────────────────────────────────────────
# C = 30, matching fig2. The original design said 50, but H(K|X) is a sum over
# cells so the counting budget grows close to exponentially in C: at C = 50 the
# pre-flight projects 0.26 bits of bias at 2^26, over the 0.2 limit, while
# C = 30 projects 0.03. Matching fig2 also makes the two figures comparable.
N_GENES = 5
N_CELLS = 30
GENES_PER_CELL = (1, 2, 3, 4, 5)
CELL_SEED = 0           # SHARED across the arms -> identical gene sets

# ─── Environment: varied per world, as in fig2 ──────────────────────────────
N_FAMILIES = 1
FAMILY_DIST = 1.0
LATENT_DIM_RANGE = (5, 9)
N_LIGANDS_RANGE = (80, 131)
FAMILY_SPREAD_RANGE = (0.08, 0.14)
MU_LIGANDS_RANGE = (2.5, 4.0)
CONC_SPAN = 1.0
CONC_STD = 1.0

MI = "kt"
MEASUREMENTS = ("grouped_counting",)   # per-epoch only; see test_*.py

EPOCHS = 5000
LR = 1e-2
BATCH_SIZE = 4096
TEST_BATCH_SIZE = 4096
COUNTING_BATCH = 2 ** 26               # what test_counting.py will use
EVAL_CHUNK_SIZE = 512
TEMPERATURE = 0.05
INIT_TEMPERATURE = 3.0
CELL_POOL_CHUNK = 128


def design():
    """One row per planned run: world x genes per cell. Independent of k_sub."""
    rows = []
    for world in range(WORLDS):
        for genes in GENES_PER_CELL:
            seed = world * 1_000_000 + genes
            sets = complete_coverage_gene_sets(N_GENES, N_CELLS, genes, seed)
            rows.append(dict(world=world, genes_per_cell=genes,
                             cell_sampling_seed=seed, gene_sets=sets,
                             distinct_gene_sets=len(set(sets)),
                             count_states=count_alphabet(sets)))
    return rows


def draw_worlds():
    """One independently drawn chemistry per world. Never seeded."""
    return dict(
        latent_dim=np.random.randint(*LATENT_DIM_RANGE, WORLDS),
        n_ligands=np.random.randint(*N_LIGANDS_RANGE, WORLDS),
        family_spread=np.random.uniform(*FAMILY_SPREAD_RANGE, WORLDS),
        mu_ligands=np.random.uniform(*MU_LIGANDS_RANGE, WORLDS),
    )


def build(k_sub, dry_run=False):
    rows = design()
    worlds = draw_worlds()
    counting_budget_report(rows, COUNTING_BATCH)

    base = dict(
        n_families=N_FAMILIES, average_family_distance=FAMILY_DIST,
        distribution_type="gaussian", observation_noise_sigma=0.0,
        n_presence_blocks=1, mu_sources=1.0, block_shared_conc_mean=False,
        n_genes=N_GENES, k_sub=k_sub, temperature=TEMPERATURE,
        kernel_params=(1.0,), initial_temperature=INIT_TEMPERATURE,
        use_interface_model=True, mi=MI, grouped=False,
        measurement_fns=MEASUREMENTS, epochs=EPOCHS, lr=LR,
        batch_size=BATCH_SIZE, test_batch_size=TEST_BATCH_SIZE,
        eval_chunk_size=EVAL_CHUNK_SIZE, per_epoch_measure=False,
        cell_stoichiometry="multinomial", cell_pool_chunk=CELL_POOL_CHUNK,
        recompute_backward=True,
    )
    runs = sweep(
        base,
        cell_gene_sets=[r["gene_sets"] for r in rows],
        cell_sampling_seed=[r["cell_sampling_seed"] for r in rows],
        latent_dim=[int(worlds["latent_dim"][r["world"]]) for r in rows],
        n_ligands=[int(worlds["n_ligands"][r["world"]]) for r in rows],
        family_spread=[float(worlds["family_spread"][r["world"]]) for r in rows],
        mu_ligands_per_source=[float(worlds["mu_ligands"][r["world"]]) for r in rows],
        conc_std=[(CONC_STD,) * int(worlds["n_ligands"][r["world"]]) for r in rows],
        conc_mean=[heterogeneous_conc_means(1, int(worlds["n_ligands"][r["world"]]),
                                            CONC_SPAN, int(np.random.randint(1 << 30)))[0]
                   for r in rows],
        # genes_per_cell is an INTEGER here and complete coverage realises it
        # exactly, so it is derivable from the config. k_sub is too. The tag is
        # the world index, which is not.
        tags=[dict(world=r["world"], distinct_gene_sets=r["distinct_gene_sets"],
                   count_states=r["count_states"]) for r in rows],
    )
    if dry_run:
        print(f"\nk_sub = {k_sub}: {len(runs)} runs, {WORLDS} world(s)\n")
        print(f"  {'w':>2} {'g':>2} {'types':>6} {'states':>10} "
              f"{'R_pool':>7} {'D':>3} {'L':>4} {'mu':>5}")
        for row, run in zip(rows, runs):
            print(f"  {row['world']:>2} {row['genes_per_cell']:>2} "
                  f"{row['distinct_gene_sets']:>6} {row['count_states']:>10,} "
                  f"{len(run.receptor_indices):>7} {run.latent_dim:>3} "
                  f"{run.n_ligands:>4} {run.mu_ligands_per_source:>5.2f}")
        return None
    return runs


def launch(k_sub, data_root, dry_run=False):
    runs = build(k_sub, dry_run)
    if runs is None:
        return
    start = time.time()
    root = Sweep(runs, name=f"fig3_k{k_sub}", data_root=data_root).execute()
    hours, rest = divmod(time.time() - start, 3600)
    minutes, seconds = divmod(rest, 60)
    print(f"\nfig3_k{k_sub}: {len(runs)} runs in {int(hours)}h {int(minutes)}m "
          f"{seconds:.0f}s -> {root}")
