#!/usr/bin/env python3
"""fig1: does splitting a fixed gene budget into more receptors buy information?

One invocation fixes n_genes and sweeps the heteromerization ratio
R / n_genes = 1..5, inside each of N_WORLDS independently drawn environments.
The figure wants n_genes in {2, 3, 5, 7, 10, 15}, so run it once per value:

    python tasks/fig1/sweep.py --n_genes 2
    python tasks/fig1/sweep.py --n_genes 3      # and so on

Ratio 1 IS the homomer condition. With strategy="cascading" the generator fills
complexity tiers in order, and tier 1 (one distinct gene) holds exactly n_genes
rings, the homomers. So R = n_genes reproduces the homomer array exactly and no
separate homomer sweep is needed.

Environments are drawn fresh on every invocation and never seeded. The claim is
about a CATEGORY of chemical worlds, so a result that needed one particular
world would not be a result. The receptor array, by contrast, IS seeded, so the
array at a given (n_genes, R) is identical across worlds and the environment is
the only thing varying.

Objective: entropy="kt", the Kolchinsky-Tracey LOWER bound on the joint entropy
H(s), which is the optimizer's target.

This trains and checkpoints only. Measure it afterwards with

    python -m tasks.fig1.test_kt
"""
import argparse
import time

import numpy as np

from model.arrays import pool_size
from train.batching import suggest
from train.config import sweep
from train.sweep import Sweep

N_WORLDS = 5                      # independent environments per invocation
K_SUB = 5
RATIOS = (1, 2, 3, 4, 5)          # R / n_genes; ratio 1 is the homomer arm


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--n_genes", type=int, required=True)
    p.add_argument("--data_root", default="data/fig1")
    p.add_argument("--gpu_gb", type=float, default=80.0,
                   help="free GPU memory used to size the batches")
    p.add_argument("--n_receptors", type=int, nargs="+", default=None,
                   help="override the receptor counts; default n_genes * 1..5")
    p.add_argument("--dry_run", action="store_true",
                   help="print the design and exit, without training anything")
    return p.parse_args()


def main():
    args = parse_args()
    receptors = args.n_receptors or [args.n_genes * r for r in RATIOS]

    # The ring pool saturates fast at small n_genes: 2 genes with k_sub=5 gives
    # only 8 distinct receptors, so R=10 is unreachable. Drop the point rather
    # than run a short array under its nominal label. The old code warned and
    # silently returned the whole pool, which put an 8-receptor run on the
    # ratio-5 curve where it duplicated the ratio-4 point.
    available = pool_size(args.n_genes, K_SUB, use_interface_model=True)
    reachable = [R for R in receptors if R <= available]
    if len(reachable) < len(receptors):
        dropped = [R for R in receptors if R > available]
        print(f"n_genes={args.n_genes}: only {available} distinct receptors exist, "
              f"so dropping R = {dropped}. The figure will have no point there.")
    if not reachable:
        raise SystemExit(f"No reachable receptor count for n_genes={args.n_genes} "
                         f"(pool is {available}).")
    receptors = reachable
    n_points = len(receptors)
    budget = int(args.gpu_gb * (1 << 30))

    # One draw per world, then repeated across that world's receptor points, so
    # the five ratios inside a world see the SAME chemistry.
    latent_dim = np.random.randint(5, 16, N_WORLDS)
    n_ligands = np.random.randint(150, 301, N_WORLDS)
    n_families = np.random.randint(5, 11, N_WORLDS)
    # family_spread is scaled by 1/sqrt(D): the volume of a D-ball grows with D,
    # so a fixed radius would mean ever sparser families as the space grows.
    family_spread = np.random.uniform(0.2, 1.0, N_WORLDS) / np.sqrt(latent_dim)
    family_distance = np.random.uniform(0.5, 1.5, N_WORLDS)
    mu_ligands = np.random.randint(20, 30, N_WORLDS)
    # Concentrations are per-ligand: each ligand has its own typical abundance.
    conc_mean = [tuple(np.random.uniform(-8.0, -3.0, n)) for n in n_ligands]

    each = lambda a: np.repeat(a, n_points).tolist()      # one value per run
    tiled = receptors * N_WORLDS

    # Batch size per point. KT costs O(B^2 * R), so the five ratios inside a
    # world need different batches; "auto" used to hide this. See doc/batching.md.
    batches = [suggest(R, "kt", n_ligands=int(L), k_sub=K_SUB, mem_budget_bytes=budget)
               for R, L in zip(tiled, each(n_ligands))]

    runs = sweep(
        dict(
            n_genes=args.n_genes,
            k_sub=K_SUB,
            temperature=0.1,
            kernel_params=(1.0,),
            distribution_type="gaussian",
            observation_noise_sigma=0.01,
            n_presence_blocks=1,
            mu_sources=1,
            block_shared_conc_mean=False,
            use_interface_model=True,
            entropy="kt",
            lr=0.05,
            per_epoch_measure=False,      # only the free training loss is logged
            compile_kt=True,              # fuse the KT tile kernel
            receptor_sampling_strategy="cascading",
            receptor_sampling_seed=0,
            # Recorded during training only if per_epoch_measure is on. The
            # reported numbers come from test_kt.py.
            measurement_fns=("entropy_kt", "entropy_kt_upper"),
        ),
        n_receptors=tiled,
        n_families=each(n_families),
        n_ligands=each(n_ligands),
        latent_dim=each(latent_dim),
        family_spread=each(family_spread),
        average_family_distance=each(family_distance),
        mu_ligands_per_source=each(mu_ligands),
        conc_mean=[cm for cm in conc_mean for _ in range(n_points)],
        conc_std=[(1.0,) * int(n) for n in each(n_ligands)],
        # Longer arrays need longer to converge; this schedule is empirical.
        epochs=[int(170 * R + 500) for R in tiled],
        batch_size=[b["batch_size"] for b in batches],
        test_batch_size=[b["test_batch_size"] for b in batches],
        # No tags: fig1's design point is the ratio R / n_genes, and both are
        # real config fields, so the figure derives it rather than trusting a
        # label that could disagree with the run.
    )

    if args.dry_run:
        print(f"fig1 ng={args.n_genes}: {len(runs)} runs, "
              f"{N_WORLDS} worlds x {n_points} receptor points\n")
        print(f"  {'world':>5s} {'R':>5s} {'ratio':>5s} {'L':>5s} {'D':>3s} "
              f"{'epochs':>7s} {'batch':>7s}")
        for i, run in enumerate(runs):
            print(f"  {i // n_points:5d} {run.n_receptors:5d} "
                  f"{run.n_receptors // args.n_genes:5d} {run.n_ligands:5d} "
                  f"{run.latent_dim:3d} {run.epochs:7d} {run.batch_size:7d}")
        return

    start = time.time()
    root = Sweep(runs, name=f"ng{args.n_genes}", data_root=args.data_root).execute()
    hours, rest = divmod(time.time() - start, 3600)
    minutes, seconds = divmod(rest, 60)
    print(f"\nfig1 ng={args.n_genes}: {len(runs)} runs in "
          f"{int(hours)}h {int(minutes)}m {seconds:.0f}s -> {root}")


if __name__ == "__main__":
    main()
