#!/usr/bin/env python3
"""large_R: homomer arrays with many more receptors than dimensions.

Companion of analytic_baseline, which tests the closed form for R <= D. Here
R >> D, at small D (2 and 3) so the regime is reachable at all. Same setting:
one Gaussian family, gaussian kernel, one ligand per sniff, homomers, shared
energies with Emax fixed and Ebase learned, near-hard readout.

Past R = D the array can no longer keep its receptors mutually
distinguishable, and the information becomes a COUNTING question: how many
response patterns can R hyperplanes cut out of a (D+1)-dimensional space?

    equal norms (shell)     ~ D log2 R
    unequal norms (ladder)  ~ (D+1) log2 R

so a shell wastes one dimension and the optimal array should SPREAD its norms.
Whether it does is what the figure checks.

lambda is set per D by lam_for(), so the geometry ratio rho = sigma sqrt(D) /
lambda is the same at every D and D is the only thing that changes.

Trains and checkpoints only. Measure with test_counting.py.

    python -m tasks.large_R.train_sweep --dry_run
    python -m tasks.large_R.train_sweep
"""
import argparse
import time

import numpy as np

from tasks._lib import large_r as LR
from tasks._lib import theory as T
from train.batching import suggest
from train.config import sweep
from train.sweep import Sweep

SIGMA = 1.0
EMAX = 10.0
CONC_MEAN = 0.0
TEMPERATURE = 0.05
N_LIGANDS = 10000
K_SUB = 5
D_DEFAULT = [2, 3]
R_DEFAULT = [2, 4, 8, 16, 32, 64]
CONC_STD_DEFAULT = [0.1086, 0.4343, 1.7372]


def theory_table(Ds, stds, Rs, emax):
    print(f"\n  {'D':>2} {'lambda':>7} {'conc_std':>9} {'nu':>7}   "
          f"{'shell bound at R=' + str(max(Rs)):>22} {'ladder':>9}")
    for D in Ds:
        lam = LR.lam_for(D, SIGMA)
        for cs in stds:
            _, sigma_c = T.from_log10(CONC_MEAN, cs)
            p = T.Params(D=D, sigma=SIGMA, lam=lam, Emax=emax, sigma_c=sigma_c)
            nu = T.lambert_optimum(p)["nu_min"]
            print(f"  {D:>2} {lam:>7.3f} {cs:>9.4f} {nu:>7.3f}   "
                  f"{LR.log2_shell_bound(max(Rs), D):>22.3f} "
                  f"{LR.log2_ladder_bound(max(Rs), D):>9.3f}")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("-D", "--latent_dim", type=int, nargs="+", default=D_DEFAULT)
    p.add_argument("--R", type=int, nargs="+", default=R_DEFAULT)
    p.add_argument("--conc_std", type=float, nargs="+", default=CONC_STD_DEFAULT)
    p.add_argument("--emax", type=float, default=EMAX)
    p.add_argument("--replicates", type=int, default=3)
    p.add_argument("--epochs_per_R", type=int, default=600)
    p.add_argument("--epochs_base", type=int, default=2000)
    p.add_argument("--data_root", default="data/large_R")
    p.add_argument("--gpu_gb", type=float, default=80.0)
    p.add_argument("--dry_run", action="store_true")
    a = p.parse_args()

    Ds, Rs, stds = a.latent_dim, a.R, a.conc_std
    blk_D = np.repeat(Ds, len(stds) * len(Rs)).tolist()
    blk_cs = np.tile(np.repeat(stds, len(Rs)), len(Ds)).tolist()
    blk_R = Rs * (len(Ds) * len(stds))
    blk_ep = [int(a.epochs_per_R * r + a.epochs_base) for r in blk_R]
    blk_lam = [(LR.lam_for(D, SIGMA),) for D in blk_D]
    rep = a.replicates
    print(f"grid: {len(Ds)} D x {len(stds)} conc_std x {len(Rs)} R x {rep} "
          f"replicates = {len(blk_R) * rep} runs")
    theory_table(Ds, stds, Rs, a.emax)

    tiled_R = blk_R * rep
    budget = int(a.gpu_gb * (1 << 30))
    batches = [suggest(R, "kt", n_ligands=N_LIGANDS, k_sub=K_SUB,
                       mem_budget_bytes=budget) for R in tiled_R]

    runs = sweep(
        dict(
            n_families=1, n_ligands=N_LIGANDS, family_spread=SIGMA,
            average_family_distance=1.0, distribution_type="gaussian",
            observation_noise_sigma=0.0, initial_temperature="auto",
            n_presence_blocks=1, mu_sources=1, mu_ligands_per_source=1e-6,
            block_shared_conc_mean=False, use_interface_model=False,
            shared_unit_energies=True, fixed_max_energy=a.emax,
            conc_mean=CONC_MEAN, k_sub=K_SUB, temperature=TEMPERATURE,
            mi="kt", grouped=False, lr=0.01, per_epoch_measure=False,
            eval_chunk_size=4096,
            measurement_fns=("codeword_entropy",),
        ),
        n_genes=tiled_R,                       # homomers: R = n_genes
        latent_dim=blk_D * rep,
        kernel_params=blk_lam * rep,
        conc_std=blk_cs * rep,
        epochs=blk_ep * rep,
        batch_size=[b["batch_size"] for b in batches],
        test_batch_size=[b["test_batch_size"] for b in batches],
        # No tags. conc_std is a config field and both counting bounds are a
        # function of (R, D), which the figure recomputes.
    )

    if a.dry_run:
        print(f"\n{len(runs)} runs\n")
        print(f"  {'D':>2} {'R':>4} {'conc_std':>9} {'lambda':>7} {'epochs':>7} "
              f"{'batch':>8} {'shell':>7} {'ladder':>7}")
        for run in runs[:len(blk_R)]:
            print(f"  {run.latent_dim:>2} {run.n_genes:>4} "
                  f"{run.conc_std:>9.4f} {run.kernel_params[0]:>7.3f} "
                  f"{run.epochs:>7} {run.batch_size:>8} "
                  f"{LR.log2_shell_bound(run.n_genes, run.latent_dim):>7.2f} "
                  f"{LR.log2_ladder_bound(run.n_genes, run.latent_dim):>7.2f}")
        print(f"  ... and {len(runs) - len(blk_R)} more (replicates)")
        return

    start = time.time()
    root = Sweep(runs, name="shell_R", data_root=a.data_root).execute()
    hours, rest = divmod(time.time() - start, 3600)
    minutes, seconds = divmod(rest, 60)
    print(f"\nlarge_R: {len(runs)} runs in {int(hours)}h {int(minutes)}m "
          f"{seconds:.0f}s -> {root}")


if __name__ == "__main__":
    main()
