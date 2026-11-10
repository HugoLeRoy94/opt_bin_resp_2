#!/usr/bin/env python3
"""two_families: does an array facing two distant families split into two arrays?

Companion of analytic_baseline (one family, R <= D) and large_R (one family,
R >> D). Same setting: gaussian kernel, one ligand per sniff, homomers, shared
energies with Emax fixed and Ebase learned, near-hard readout.

Two families appear with probabilities pi and 1 - pi, placed far enough apart
(average_family_distance = 20 at sigma = 1, D = 8) that a receptor tuned to one
never fires for a ligand of the other. If that holds, each receptor serves
exactly one family and

    MI_2 ~ H(F) + max_{R_1} [ pi G(R_1) + (1-pi) G(R-R_1) ] - H(F|A)

with G(n) the MI of an optimal single-family array of n receptors.

REGIME. D = 8, lambda = 4, conc_std = 0.2171 (nu = 0.67, no saturation). There a
single family is near-linear, G(n) ~ n, up to n = D + 1 = 9, and bends past it.
Two families can host two simplices, so up to R = 18 the pair stays near-linear
per family, MI_2 ~ 1 + G(R/2), against G(R) for one family. That gap is a few
bits, against about 0.7 bit in the old D = 2, nu = 1.7 regime, where G was
already logarithmic and splitting cost almost nothing.

pi = 1/2. With unequal pi the rare family gets no receptor until
pi dG(R_1) < (1-pi) G(1), which at D = 8 happens near R_1 ~ 20-30, beyond the
measured G (n <= 16) and close to the counting ceiling.

G IS NOT MEASURED HERE. It comes from the analytic_baseline sweep at D = 8,
lambda = 4, conc_std = 0.2171, whose parameters this task copies, so
analytic_baseline must be run and measured first. Its R grid stops at 16, which
covers R_f = R/2 up to R = 32.

Trains and checkpoints only. Measure with test_counting.py.

    python -m tasks.two_families.train_sweep --dry_run
    python -m tasks.two_families.train_sweep
"""
import argparse
import time

from tasks._lib import theory as T
from tasks._lib.two_fam import h2
from train.batching import suggest
from train.config import sweep
from train.sweep import Sweep

D = 8
SIGMA = 1.0
LAMBDA = 4.0                   # rho = sigma sqrt(D) / lambda = 0.707
EMAX = 10.0
CONC_MEAN = 0.0
CONC_STD = 0.2171              # matches analytic_baseline, so its G applies here
TEMPERATURE = 0.05
N_LIGANDS = 10000
K_SUB = 5
PI_DEFAULT = 0.5
# Centres are drawn at random in a ball. In D = 8 their distance concentrates:
# 5% of draws fall below 0.65 x the average, here 13, against about 8 for one
# receptor's reach (U* + field radius) plus the other cloud. Farther still would
# leave the centres ~15 from the origin, where the receptors start.
FAMILY_DIST_DEFAULT = 20.0
R_DEFAULT = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 24, 28, 32]


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

    lam = LAMBDA
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
        # No tags. pi is family_weights[0], lambda is kernel_params[0], the
        # family distance is a config field, and nu is derived from them.
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
