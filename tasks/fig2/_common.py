"""Everything the four fig2 arms share. Each arm script states only what it is.

fig2 asks: as a cell expresses more genes, does the information its population
carries keep rising, or does it saturate? Four arms cross two binary choices.

  HOW GENE SETS ARE DRAWN
    complete   every gene is guaranteed to appear in at least one cell
    bernoulli  each gene independent at probability p; a gene can end up nowhere

  WHAT A CELL BUILDS FROM ITS GENES
    heteromers every receptor its genes allow, mixed subunits included
    homomers   single-gene receptors only, stated outright as cell_receptors

Arms with the same strategy share gene sets, because the repertoire is the
DESIGN and holding it fixed is what makes heteromers-vs-homomers a comparison
of what cells can build. Environments are NOT shared: each arm draws its own.

WORLDS is the one number to raise for more replicates. A world is an
independently drawn chemistry, not a redraw inside one, so the spread across
worlds is what belongs on an error bar.
"""
import time

import numpy as np

from tasks._lib.budget import counting_budget_report
from tasks._lib.repertoires import (bernoulli_gene_sets, complete_coverage_gene_sets_mixed,
                                    count_alphabet, heterogeneous_conc_means, homomers_only)
from train.config import sweep
from train.sweep import Sweep

WORLDS = 1              # raise this for more replicates; nothing else changes

# ─── The design grid ────────────────────────────────────────────────────────
# C = 50 is deliberately absent. H(K|X) is a sum over cells and reaches ~15 bits
# at C = 50, needing a budget of 2^30 (about 345 h) to stay under 0.2 bits of
# bias. The pre-flight below refuses to launch if you put it back.
N_GENES_LIST = (2, 3, 4, 5)
N_CELLS_LIST = (20, 30)
GENE_STEP = 0.25
K_SUB = 5

# ─── Environment: varied per world, around fig2's original constants ────────
# n_families stays 1: fig2 has no family structure and varying it changes the
# question. average_family_distance stays 1.0: with one family it only offsets
# the cloud from the receptor initialisation, so varying it adds noise, not
# chemistry.
N_FAMILIES = 1
FAMILY_DIST = 1.0
LATENT_DIM_RANGE = (5, 9)           # was 6
N_LIGANDS_RANGE = (80, 131)         # was 100
FAMILY_SPREAD_RANGE = (0.08, 0.14)  # was 0.10
MU_LIGANDS_RANGE = (2.5, 4.0)       # was 3.0; ligands present per sniff
CONC_SPAN = 1.0                     # log10 mean in +/- 1 -> 2 decades
CONC_STD = 1.0

# ─── Objective and measurement ──────────────────────────────────────────────
# kt_mi trains; grouped_counting is the only measurement and already carries
# MI, H(Y), H(Y|X) and alphabet coverage as ONE estimator family, so
# MI = H(Y) - H(Y|X) holds exactly.
ENTROPY = "kt_mi"
# Recorded during training only when per_epoch_measure is on, which it is not.
# The reported numbers come from test_counting.py and test_kt.py.
MEASUREMENTS = ("grouped_counting",)

# ─── Training ───────────────────────────────────────────────────────────────
EPOCHS = 5000
LR = 1e-2
BATCH_SIZE = 4096
TEST_BATCH_SIZE = 4096
# Budget the counting pass will use. Not a config field: test_counting.py owns
# it. Kept here because the pre-flight below has to check the same number.
FINAL_BATCH_SIZE = 2 ** 26          # 2^20 made C=50 go NEGATIVE
EVAL_CHUNK_SIZE = 512
TEMPERATURE = 0.05
INIT_TEMPERATURE = 3.0
CELL_POOL_CHUNK = 128


def targets(n_genes):
    """Mean genes per cell, from 1 to n_genes in GENE_STEP increments."""
    return [round(float(t), 4)
            for t in np.arange(1.0, n_genes + 1e-9, GENE_STEP)]


def design(strategy):
    """One row per planned run: world x G x C x mean genes per cell.

    The gene-set seed is a function of (world, G, C, target) only, so the two
    arms sharing a strategy get identical repertoires.
    """
    rows = []
    for world in range(WORLDS):
        for n_genes in N_GENES_LIST:
            for n_cells in N_CELLS_LIST:
                for target in targets(n_genes):
                    seed = (world * 1_000_000 + n_genes * 10_000
                            + n_cells * 100 + int(round(target * 4)))
                    if strategy == "complete":
                        sets, realised = complete_coverage_gene_sets_mixed(
                            n_genes, n_cells, target, seed)
                        probability = float("nan")
                    else:
                        sets, probability, realised = bernoulli_gene_sets(
                            n_genes, n_cells, target, seed)
                    rows.append(dict(
                        world=world, n_genes=n_genes, n_cells=n_cells,
                        target_genes=target, realised_genes=realised,
                        gene_probability=probability, cell_sampling_seed=seed,
                        gene_sets=sets, distinct_gene_sets=len(set(sets)),
                        genes_seen=len({g for gs in sets for g in gs}),
                        count_states=count_alphabet(sets)))
    return rows


def draw_worlds():
    """One independently drawn chemistry per world. Never seeded.

    The claim is about a CATEGORY of chemical worlds, so a result that needed
    one particular world would not be a result.
    """
    return dict(
        latent_dim=np.random.randint(*LATENT_DIM_RANGE, WORLDS),
        n_ligands=np.random.randint(*N_LIGANDS_RANGE, WORLDS),
        family_spread=np.random.uniform(*FAMILY_SPREAD_RANGE, WORLDS),
        mu_ligands=np.random.uniform(*MU_LIGANDS_RANGE, WORLDS),
    )


def build(strategy, condition, data_root, dry_run=False):
    """The runs for one arm, after the pre-flight budget check."""
    rows = design(strategy)
    worlds = draw_worlds()
    per_world = [w for w in worlds["n_ligands"] for _ in range(len(rows) // WORLDS)]

    # Refuses to launch if sampled counting cannot resolve this design.
    counting_budget_report(rows, FINAL_BATCH_SIZE)

    base = dict(
        n_families=N_FAMILIES, average_family_distance=FAMILY_DIST,
        distribution_type="gaussian", observation_noise_sigma=0.0,
        n_presence_blocks=1, mu_sources=1.0, block_shared_conc_mean=False,
        k_sub=K_SUB, temperature=TEMPERATURE, kernel_params=(1.0,),
        initial_temperature=INIT_TEMPERATURE, use_interface_model=True,
        entropy=ENTROPY, measurement_fns=MEASUREMENTS,
        epochs=EPOCHS, lr=LR, batch_size=BATCH_SIZE,
        test_batch_size=TEST_BATCH_SIZE,
        eval_chunk_size=EVAL_CHUNK_SIZE, per_epoch_measure=False,
        cell_stoichiometry="multinomial", cell_pool_chunk=CELL_POOL_CHUNK,
        recompute_backward=True,
    )
    axes = dict(
        n_genes=[r["n_genes"] for r in rows],
        n_cells=[r["n_cells"] for r in rows],
        cell_sampling_seed=[r["cell_sampling_seed"] for r in rows],
        latent_dim=[int(worlds["latent_dim"][r["world"]]) for r in rows],
        n_ligands=[int(worlds["n_ligands"][r["world"]]) for r in rows],
        family_spread=[float(worlds["family_spread"][r["world"]]) for r in rows],
        mu_ligands_per_source=[float(worlds["mu_ligands"][r["world"]]) for r in rows],
        conc_std=[(CONC_STD,) * int(worlds["n_ligands"][r["world"]]) for r in rows],
        conc_mean=[heterogeneous_conc_means(1, int(worlds["n_ligands"][r["world"]]),
                                            CONC_SPAN, int(np.random.randint(1 << 30)))[0]
                   for r in rows],
        # Labels, not physics. target_genes is the one the figure GROUPS by: the
        # config records the gene sets a cell has, never the mean that was asked
        # for, and under the bernoulli strategy the realised mean drifts from it
        # by more than the spacing between targets. The realised value is
        # derived at load time as genes_per_cell, so it is not tagged.
        tags=[dict(strategy=strategy, condition=condition, world=r["world"],
                   target_genes=r["target_genes"],
                   gene_probability=r["gene_probability"],
                   distinct_gene_sets=r["distinct_gene_sets"],
                   genes_seen=r["genes_seen"], count_states=r["count_states"])
              for r in rows],
    )
    if condition == "homomers":
        # Stated outright: the gene-set path cannot express "homomers only",
        # because expressing u and v necessarily also builds every mix.
        axes["cell_receptors"] = [homomers_only(r["gene_sets"], K_SUB) for r in rows]
    else:
        axes["cell_gene_sets"] = [r["gene_sets"] for r in rows]

    runs = sweep(base, **axes)
    if dry_run:
        _report(strategy, condition, rows, runs)
        return None
    return runs


def _report(strategy, condition, rows, runs):
    print(f"\n{strategy}/{condition}: {len(runs)} runs, {WORLDS} world(s)\n")
    print(f"  {'w':>2} {'G':>2} {'C':>4} {'target':>7} {'realised':>9} "
          f"{'types':>6} {'seen':>6} {'states':>8} {'D':>3} {'L':>4} {'mu':>5}")
    for row, run in zip(rows, runs):
        print(f"  {row['world']:>2} {row['n_genes']:>2} {row['n_cells']:>4} "
              f"{row['target_genes']:>7.2f} {row['realised_genes']:>9.2f} "
              f"{row['distinct_gene_sets']:>6} "
              f"{row['genes_seen']:>3}/{row['n_genes']:<2} {row['count_states']:>8,} "
              f"{run.latent_dim:>3} {run.n_ligands:>4} {run.mu_ligands_per_source:>5.2f}")


def launch(strategy, condition, data_root, dry_run=False):
    """Build and run one arm. The sweep name matches the calling script."""
    name = f"fig2_{strategy}_{condition}"
    runs = build(strategy, condition, data_root, dry_run)
    if runs is None:
        return
    start = time.time()
    root = Sweep(runs, name=name, data_root=data_root).execute()
    hours, rest = divmod(time.time() - start, 3600)
    minutes, seconds = divmod(rest, 60)
    print(f"\n{name}: {len(runs)} runs in {int(hours)}h {int(minutes)}m "
          f"{seconds:.0f}s -> {root}")
