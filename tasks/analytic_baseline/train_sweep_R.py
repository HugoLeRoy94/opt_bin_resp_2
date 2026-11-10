#!/usr/bin/env python3
"""analytic_baseline: a homomer array against the closed form, sweeping R at fixed D.

Reference: analytics/homomer_array_mi.tex. The closed form assumes exactly the
setting built here: one Gaussian ligand family, a saturating (gaussian) kernel,
exactly one ligand per sniff, homomers only, and a near-hard threshold.

Two axes, and they do different things.

  R = n_genes   homomers give one receptor per gene, so sweeping n_genes sweeps
                R. Crossing R = D moves the array from the exchangeable regime
                (MI ~ R eta) into geometric saturation (MI ~ D log2 R).
                THIS AXIS DOES NOT MOVE nu.

  conc_std      nu is independent of R. It is set by the geometry ratio
                rho = sigma sqrt(D) / lambda and the dynamic range
                Gamma = Emax / sigma_c. lambda has a floor, since nu_min is
                non-monotone in it, so the clean knob for nu is the
                concentration spread. Each conc_std is one nu.

A6 of the analytics: ONE learnable (Ebase, Emax) pair for the whole array
(shared_unit_energies). Without it the simulator optimises n_genes of each,
which is a different model from the closed form and the comparison would hold
only where the optimiser happened to make them agree.

Trains and checkpoints only. Measure with test_counting.py and test_kt.py.

    python -m tasks.analytic_baseline.train_sweep_R --dry_run
    python -m tasks.analytic_baseline.train_sweep_R
"""
import argparse
import time

import numpy as np

from tasks._lib import theory as T
from train.batching import suggest
from train.config import sweep
from train.sweep import Sweep

D_DEFAULT = 8
SIGMA = 1.0            # family_spread: per-coordinate spread of the ligand cloud
LAMBDA = 4.0           # kernel range; rho = sigma sqrt(D) / lambda = 0.707 at D=8
EMAX = 10.0            # fixed E_max, natural-log units
CONC_MEAN = 0.0        # log10; only sets the Ebase init, which is learnable
# Four values of nu. The simulator stores conc_std in log10 while the analytics
# is in natural log, so sigma_c = conc_std * ln10. Getting this wrong rescales
# nu by 2.3; see theory.from_log10.
CONC_STD_DEFAULT = [1.7372, 0.8686, 0.4343, 0.2171]
R_DEFAULT = [2, 3, 4, 5, 6, 8, 10, 12, 14, 16]
TEMPERATURE = 0.05     # needs T << 0.1 * Delta_priv ~ 0.11
N_LIGANDS = 10000
K_SUB = 5


def theory_table(D, lam, stds, emax):
    """What the closed form expects, before a single GPU hour is spent."""
    print(f"\n  {'conc_std':>9} {'sigma_c':>8} {'rho':>6} {'Gamma':>7} "
          f"{'nu_min':>7} {'eta':>7} {'selfcons':>9}")
    for cs in stds:
        _, sigma_c = T.from_log10(CONC_MEAN, cs)
        p = T.Params(D=D, sigma=SIGMA, lam=lam, Emax=emax, sigma_c=sigma_c)
        opt = T.lambert_optimum(p)
        print(f"  {cs:>9.4f} {sigma_c:>8.3f} {p.rho:>6.3f} {p.Gamma:>7.2f} "
              f"{opt['nu_min']:>7.3f} {T.eta(opt['nu_min']):>7.4f} "
              f"{opt['selfcons']:>9.2f}")
    print("  selfcons > 3 is where the delta-method route is claimed to hold.")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("-D", "--latent_dim", type=int, default=D_DEFAULT)
    p.add_argument("--lam", type=float, default=LAMBDA)
    p.add_argument("--R", type=int, nargs="+", default=R_DEFAULT)
    p.add_argument("--conc_std", type=float, nargs="+", default=CONC_STD_DEFAULT)
    p.add_argument("--replicates", type=int, default=3,
                   help="independent environments per design point")
    p.add_argument("--epochs_per_R", type=int, default=600)
    p.add_argument("--epochs_base", type=int, default=2000)
    p.add_argument("--emax", type=float, default=EMAX)
    p.add_argument("--data_root", default="data/analytic_baseline")
    p.add_argument("--gpu_gb", type=float, default=80.0)
    p.add_argument("--dry_run", action="store_true")
    return p.parse_args()


def main():
    a = parse_args()
    D, lam, Rs, stds = a.latent_dim, a.lam, a.R, a.conc_std
    print(f"grid: {len(Rs)} R x {len(stds)} conc_std x {a.replicates} replicates "
          f"= {len(Rs) * len(stds) * a.replicates} runs")
    print(f"R crosses D = {D} at R = {D}: {sum(r <= D for r in Rs)} below, "
          f"{sum(r > D for r in Rs)} above")
    theory_table(D, lam, stds, a.emax)

    # Cartesian block, R inner and conc_std outer, then replicated. The global
    # RNG advances run to run, so each replicate is an independent environment
    # and no seed is shared between conditions.
    blk_R = Rs * len(stds)
    blk_cs = np.repeat(stds, len(Rs)).tolist()
    blk_ep = [int(a.epochs_per_R * r + a.epochs_base) for r in Rs] * len(stds)
    tiled_R = blk_R * a.replicates
    budget = int(a.gpu_gb * (1 << 30))
    batches = [suggest(R, "kt", n_ligands=N_LIGANDS, k_sub=K_SUB,
                       mem_budget_bytes=budget) for R in tiled_R]

    runs = sweep(
        dict(
            n_families=1, n_ligands=N_LIGANDS, latent_dim=D,
            family_spread=SIGMA, average_family_distance=1.0,
            distribution_type="gaussian",
            observation_noise_sigma=0.0,          # A2: the sniff IS the ligand
            initial_temperature="auto",
            # Exactly one ligand per sniff (A2).
            n_presence_blocks=1, mu_sources=1, mu_ligands_per_source=1e-6,
            block_shared_conc_mean=False,
            use_interface_model=False,
            shared_unit_energies=True,            # A6, see the docstring
            fixed_max_energy=a.emax,
            conc_mean=CONC_MEAN,
            k_sub=K_SUB, temperature=TEMPERATURE, kernel_params=(lam,),
            mi="kt", grouped=False, lr=0.01, per_epoch_measure=False,
            eval_chunk_size=4096,
            measurement_fns=("codeword_entropy",),
        ),
        n_genes=tiled_R,                          # homomers: R = n_genes
        conc_std=blk_cs * a.replicates,
        epochs=blk_ep * a.replicates,
        batch_size=[b["batch_size"] for b in batches],
        test_batch_size=[b["test_batch_size"] for b in batches],
        # nu is the scalar the theory collapses onto, and it is not a config
        # field: it comes from (D, sigma, lambda, Emax, sigma_c). Tag it so the
        # figure can group by it without recomputing the closed form.
        tags=[dict(nu=round(float(T.lambert_optimum(T.Params(
                        D=D, sigma=SIGMA, lam=lam, Emax=a.emax,
                        sigma_c=T.from_log10(CONC_MEAN, cs)[1]))["nu_min"]), 6),
                   conc_std_log10=cs, lam=lam)
              for cs in blk_cs * a.replicates],
    )

    if a.dry_run:
        print(f"\n{len(runs)} runs\n")
        print(f"  {'R':>3} {'conc_std':>9} {'nu':>7} {'epochs':>7} {'batch':>8}")
        for run in runs[:len(Rs) * len(stds)]:
            print(f"  {run.n_genes:>3} {run.tags['conc_std_log10']:>9.4f} "
                  f"{run.tags['nu']:>7.3f} {run.epochs:>7} {run.batch_size:>8}")
        print(f"  ... and {len(runs) - len(Rs) * len(stds)} more (replicates)")
        return

    start = time.time()
    root = Sweep(runs, name="homomer_R", data_root=a.data_root).execute()
    hours, rest = divmod(time.time() - start, 3600)
    minutes, seconds = divmod(rest, 60)
    print(f"\nanalytic_baseline: {len(runs)} runs in {int(hours)}h "
          f"{int(minutes)}m {seconds:.0f}s -> {root}")


if __name__ == "__main__":
    main()
