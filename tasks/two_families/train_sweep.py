#!/usr/bin/env python3
"""two_families: does an array facing two distant families split into two arrays?

Companion of analytic_baseline (one family, R <= D) and large_R (one family,
R >> D). Same setting: gaussian kernel, one ligand per sniff, homomers, shared
energies with Emax fixed and Ebase learned, near-hard readout.

Two families appear with probabilities pi and 1 - pi, placed far enough apart
(average_family_distance = 8 at sigma = 1) that a receptor tuned to one never
fires for a ligand of the other. If that holds, each receptor serves exactly
one family and

    MI_2 ~ H(F) + max_{R_1} [ pi G(R_1) + (1-pi) G(R-R_1) ] - H(F|A)

with G(n) the MI of an optimal single-family array of n receptors.

G IS NOT MEASURED HERE. It comes from the large_R sweep at D = 2,
conc_std = 1.7372, whose parameters this task deliberately copies, so
large_R must be run and measured before this task can be analysed.

Trains and checkpoints only. Measure with test_counting.py.

    python -m tasks.two_families.train_sweep --dry_run
    python -m tasks.two_families.train_sweep
"""
import argparse
import time

from tasks._lib import large_r as LR
from tasks._lib import theory as T
from tasks._lib.two_fam import h2
from train.batching import suggest
from train.config import sweep
from train.sweep import Sweep

D = 2
SIGMA = 1.0
EMAX = 10.0
CONC_MEAN = 0.0
CONC_STD = 1.7372              # matches large_R, so its G applies here
TEMPERATURE = 0.05
N_LIGANDS = 10000
K_SUB = 5
PI_DEFAULT = 0.8
FAMILY_DIST_DEFAULT = 8.0      # centres far apart: no receptor serves both
R_DEFAULT = [2, 3, 4, 6, 8, 10, 12, 16, 24, 32, 48, 64]


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--pi", type=float, default=PI_DEFAULT,
                   help="probability of family 1; family 2 gets 1 - pi")
    p.add_argument("--family_dist", type=float, default=FAMILY_DIST_DEFAULT)
    p.add_argument("--R", type=int, nargs="+", default=R_DEFAULT)
    p.add_argument("--replicates", type=int, default=1)
    p.add_argument("--epochs_per_R", type=int, default=600)
    p.add_argument("--epochs_base", type=int, default=2000)
    p.add_argument("--emax", type=float, default=EMAX)
    p.add_argument("--data_root", default="data/two_families")
    p.add_argument("--gpu_gb", type=float, default=80.0)
    p.add_argument("--dry_run", action="store_true")
    a = p.parse_args()

    lam = LR.lam_for(D, SIGMA)
    _, sigma_c = T.from_log10(CONC_MEAN, CONC_STD)
    params = T.Params(D=D, sigma=SIGMA, lam=lam, Emax=a.emax, sigma_c=sigma_c)
    nu = T.lambert_optimum(params)["nu_min"]
    tiled_R = a.R * a.replicates
    print(f"grid: {len(a.R)} R x {a.replicates} replicates = {len(tiled_R)} runs")
    print(f"  D = {D}, lambda = {lam:.3f}, nu = {nu:.3f}, pi = {a.pi}, "
          f"family distance = {a.family_dist}")
    print(f"  H(F) = {h2(a.pi):.4f} bits, the most the family label alone can carry")

    budget = int(a.gpu_gb * (1 << 30))
    batches = [suggest(R, "kt", n_ligands=N_LIGANDS, k_sub=K_SUB,
                       mem_budget_bytes=budget) for R in tiled_R]

    runs = sweep(
        dict(
            n_families=2, n_ligands=N_LIGANDS, latent_dim=D,
            family_spread=SIGMA, average_family_distance=a.family_dist,
            # The whole premise: family 1 is seen pi of the time.
            family_weights=(a.pi, round(1.0 - a.pi, 12)),
            distribution_type="gaussian", observation_noise_sigma=0.0,
            initial_temperature="auto",
            n_presence_blocks=1, mu_sources=1, mu_ligands_per_source=1e-6,
            block_shared_conc_mean=False, use_interface_model=False,
            shared_unit_energies=True, fixed_max_energy=a.emax,
            conc_mean=CONC_MEAN, conc_std=CONC_STD,
            k_sub=K_SUB, temperature=TEMPERATURE, kernel_params=(lam,),
            mi="kt", grouped=False, lr=0.01, per_epoch_measure=False,
            eval_chunk_size=4096,
            measurement_fns=("codeword_entropy",),
        ),
        n_genes=tiled_R,                       # homomers: R = n_genes
        epochs=[int(a.epochs_per_R * R + a.epochs_base) for R in tiled_R],
        batch_size=[b["batch_size"] for b in batches],
        test_batch_size=[b["test_batch_size"] for b in batches],
        tags=[dict(pi=a.pi, nu=round(float(nu), 6), lam=lam,
                   family_dist=a.family_dist) for _ in tiled_R],
    )

    if a.dry_run:
        print(f"\n{len(runs)} runs\n")
        print(f"  {'R':>4} {'epochs':>7} {'batch':>8}")
        for run in runs:
            print(f"  {run.n_genes:>4} {run.epochs:>7} {run.batch_size:>8}")
        return

    start = time.time()
    root = Sweep(runs, name=f"two_fam_pi{a.pi:g}", data_root=a.data_root).execute()
    hours, rest = divmod(time.time() - start, 3600)
    minutes, seconds = divmod(rest, 60)
    print(f"\ntwo_families: {len(runs)} runs in {int(hours)}h {int(minutes)}m "
          f"{seconds:.0f}s -> {root}")


if __name__ == "__main__":
    main()
