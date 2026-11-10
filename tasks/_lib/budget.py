"""Projected bias of sampled grouped counting, BEFORE you spend the cluster time.

Sampled counting estimates H(K) from B draws, so it cannot exceed log2(B), while
H(K|X) is computed analytically and has no such cap. As H(K|X) approaches
log2(B) the reported MI = H(K)hat - H(K|X) collapses, and past it the MI goes
NEGATIVE. That is arithmetic, not a subtle bias, and it has happened twice in
this project: fig2 at C = 50, and the whole dilution task.

The fix is to find out before training. Call counting_budget_report on the
design and refuse to launch if the projection is too large.

Calibration, from estimator_crosscheck measuring sampled counting against exact
enumeration over 12 points of the G=5, C=30 arrays:

    bias [bits] ~= 3.34 * (B / 2^H(K)) ** -0.92

so what matters is the ratio of budget to EFFECTIVE ALPHABET 2^H(K), not the
budget alone. Bias falls below 0.1 bits at a ratio near 46.
"""
import math
from collections import Counter

__all__ = ["projected_counting_bias", "counting_budget_report"]

# Per-cell firing rate reproducing the observed H(Y|X) per cell, about 0.4 bits
# once a cell holds more than one receptor.
FIRING_P = 0.0905

# Stand-in for the MI a trained array will reach. It is NOT a config field and
# never reaches the simulation: it only lets this module predict
# H(K) = MI + H(K|X) before the MI exists.
#
# It is an UPPER BOUND, not a typical value. Setting it too LOW underestimates
# H(K), hence the budget, and it fails worst on the best-performing arrays,
# which are exactly the ones whose alphabet was most underestimated. Setting it
# too HIGH only over-provisions. The errors are not symmetric, so err high.
#
# Largest MI observed across the kept fig2 runs: 8.69 bits.
MAX_MI = 10.0


def _binomial_entropy(n, p):
    terms = [math.comb(n, k) * p ** k * (1 - p) ** (n - k) for k in range(n + 1)]
    return -sum(q * math.log2(q) for q in terms if q > 0)


def projected_counting_bias(gene_sets, batch_size, firing_p=FIRING_P, max_mi=MAX_MI):
    """(bias in bits, H(K|X), H(K)) for one design point.

    Cells with an IDENTICAL gene set are one group, and a group of n contributes
    H(Binom(n, p)) rather than n * H(p). That is exactly why grouping decides
    whether counting is usable at all.
    """
    if batch_size < 1:
        raise ValueError("batch_size must be positive.")
    sizes = Counter(tuple(gs) for gs in gene_sets).values()
    h_kx = sum(_binomial_entropy(n, firing_p) for n in sizes)
    h_k = max_mi + h_kx
    return 3.34 * (batch_size / 2 ** h_k) ** -0.92, h_kx, h_k


def counting_budget_report(rows, batch_size, key="gene_sets", limit=0.2,
                           firing_p=FIRING_P, max_mi=MAX_MI):
    """Print the projection per design point and RAISE if the worst exceeds `limit`.

    Raising matters more than printing. Above about 0.2 bits the bias DIFFERS
    between design points, so it distorts the SHAPE of a curve rather than
    shifting it, and no amount of staring at the figure afterwards reveals that.
    """
    print(f"\n{'point':>5} {'cells':>6} {'groups':>7} {'H(K|X)':>8} {'H(K)~':>7} "
          f"{'bias':>7} {'B for 0.1b':>14}")
    worst = 0.0
    for index, row in enumerate(rows):
        bias, h_kx, h_k = projected_counting_bias(row[key], batch_size, firing_p, max_mi)
        worst = max(worst, bias)
        print(f"{index:>5} {len(row[key]):>6} {len(set(map(tuple, row[key]))):>7} "
              f"{h_kx:>8.1f} {h_k:>7.1f} {bias:>6.2f}b {46 * 2 ** h_k:>14,.0f}")
    print(f"\nbudget {batch_size:,} = 2^{math.log2(batch_size):.0f}; "
          f"worst projected bias {worst:.2f} bits (MAX_MI = {max_mi})")
    if worst > limit:
        raise ValueError(
            f"Projected counting bias {worst:.2f} bits exceeds {limit}. Above this "
            f"the bias differs BETWEEN design points, so it distorts the curve "
            f"shape. Raise the test script's batch size, or cut n_cells: H(K|X) is a sum "
            f"over cells, so 2^H(K) grows close to exponentially in C while an "
            f"affordable budget grows linearly.")
    return worst
