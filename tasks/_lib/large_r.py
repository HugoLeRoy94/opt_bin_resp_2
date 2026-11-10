"""Counting bounds for homomer arrays with many more receptors than dimensions.

Companion of the closed form in theory.py, which holds for R <= D. Here R >> D,
so the array can no longer keep its receptors mutually distinguishable and the
information is set by COUNTING how many response patterns R hyperplanes can cut
out of the stimulus space.

With equal norms U, receptor r fires iff u_r . omega > c, where omega = l/||l||
is the ligand direction and c = tau/||l|| the shared threshold over the ligand
norm. The pattern depends on (omega, c) only: it is the set of receptor
directions inside a spherical cap centred on omega of angular radius arccos c.

Each receptor is then a hyperplane in (l, tau), a space of dimension D+1:

    equal norms (shell)    all hyperplanes pass through one point
                           log2(2 sum_{i<=D} C(R-1, i))  ->  ~ D log2 R
    unequal norms (ladder) hyperplanes in general position
                           log2(sum_{i<=D+1} C(R, i))    ->  ~ (D+1) log2 R

So a shell WASTES one dimension at large R, and the optimal array should spread
its norms rather than sit on a sphere.
"""
import math

import numpy as np

RHO = math.sqrt(0.5)          # sigma sqrt(D) / lambda, as in analytic_baseline


RHO = math.sqrt(0.5)          # sigma sqrt(D) / lambda, as in analytic_baseline


def lam_for(D: int, sigma: float = 1.0, rho: float = RHO) -> float:
    """Kernel range giving the same geometry ratio rho at every D."""
    return sigma * math.sqrt(D) / rho


def log2_shell_bound(R: int, D: int) -> float:
    """Equal norms: R hyperplanes through one point of R^{D+1}."""
    return math.log2(2 * sum(math.comb(R - 1, i) for i in range(D + 1)))


def log2_ladder_bound(R: int, D: int) -> float:
    """Unequal norms: R hyperplanes in general position in R^{D+1}."""
    return math.log2(sum(math.comb(R, i) for i in range(D + 2)))


def c_law(p, U: float, ebase: float, n: int = 400_000, seed: int = 0) -> dict:
    """Law of c = tau/||l|| for a shell of radius U: interior sd and atom weights."""
    rng = np.random.default_rng(seed)
    l2 = p.sigma ** 2 * rng.chisquare(p.D, n)
    s = p.m + p.sigma_c * rng.standard_normal(n)
    arg = 1.0 - (s - ebase) / p.Emax
    with np.errstate(divide="ignore", invalid="ignore"):
        B = np.where(arg > 0, -p.lam ** 2 * np.log(arg), np.inf)
    c = (l2 - B + U ** 2) / (2 * U * np.sqrt(l2))
    inside = np.abs(c) < 1
    return {"sd_c": float(c[inside].std()), "mean_c": float(c[inside].mean()),
            "p_on": float(np.mean(c <= -1)), "p_off": float(np.mean(c >= 1))}
