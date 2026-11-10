# %%
"""analytic_baseline: measured MI against the closed form, across the R = D crossover.

Reference: analytics/homomer_array_mi.tex.

The closed form predicts that below R = D the array is exchangeable and the
information grows linearly, MI = R eta(nu), while above it the receptors can no
longer stay mutually distinguishable and MI saturates toward D log2 R. One
curve per nu, which the concentration spread sets.

Observation noise is zero and the threshold is near-hard, so H(Y|X) is about 0
and the hard-codeword entropy IS the mutual information. That is the measured
quantity plotted here.

Two ceilings matter and both are drawn. A plug-in count estimate cannot exceed
log2(B) however good the array is, and the missing mass says how much
probability was never sampled. Where a point approaches either, it is reporting
the measurement budget rather than the array, and the KT bracket is the one to
believe.
"""
import matplotlib.pyplot as plt
import numpy as np

from analysis.load import across_replicates, runs
from tasks._lib import theory as T

GOAL = "analytic_baseline"
MI = "codeword_entropy_mm"           # Miller-Madow corrected plug-in
MI_PLUGIN = "codeword_entropy_plugin"
MISS = "codeword_entropy_missing_mass"
CEIL = "codeword_entropy_log2B"
KT_LO, KT_UP = "mutual_information_kt", "mutual_information_kt_upper"

frame = runs(GOAL, latest_only=True)
points = across_replicates(frame, keys=["nu", "n_genes"])
nus = sorted(points.nu.unique())
colours = plt.colormaps["viridis"](np.linspace(0, .9, len(nus)))

# %%
fig, ax = plt.subplots(figsize=(6.5, 4.8))
D = int(frame.latent_dim.iloc[0])
for colour, nu in zip(colours, nus):
    curve = points[points.nu == nu].sort_values("n_genes")
    R = curve.n_genes.to_numpy()
    ax.plot(R, curve[MI], "o-", color=colour, label=f"nu = {nu:.2f}")
    ax.fill_between(R, curve[f"{MI} lo"], curve[f"{MI} hi"],
                    color=colour, alpha=.15, linewidth=0)
    # The closed form, on the same axis. Eq. 48 below the crossover.
    ax.plot(R, [T.mi_predicted(nu, int(r))for r in R], "--",
            color=colour, linewidth=1, alpha=.8)

ax.axvline(D, color="k", lw=.8, ls=":")
ax.annotate(f"R = D = {D}", (D, ax.get_ylim()[1]), xytext=(4, -12),
            textcoords="offset points", fontsize=8)
ax.plot(sorted(points.n_genes.unique()), points.groupby("n_genes")[CEIL].first(),
        ":", color="0.4", lw=1, label="log2(B), the counting ceiling")
ax.set(xlabel="receptors R (= genes, homomers)", ylabel="MI [bits]",
       title="solid: measured codewords   dashed: closed form   band: 1 sd across worlds")
ax.legend(fontsize=8)
plt.tight_layout()
plt.show()

# %%
# ── is each point inside its measurement budget? ────────────────────────────
# Missing mass rising means the alphabet outran the samples, so the plug-in is
# biased DOWN. Comparing against the KT bracket says whether that has bitten.
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
for colour, nu in zip(colours, nus):
    curve = points[points.nu == nu].sort_values("n_genes")
    axes[0].plot(curve.n_genes, curve[MISS], "o-", color=colour, label=f"nu = {nu:.2f}")
    if KT_LO in curve:
        axes[1].plot(curve.n_genes, curve[MI], "o-", color=colour)
        axes[1].fill_between(curve.n_genes, curve[KT_LO], curve[KT_UP],
                             color=colour, alpha=.15, linewidth=0)
axes[0].set(xlabel="receptors R", ylabel="missing mass",
            title="probability never sampled", yscale="log")
axes[0].legend(fontsize=8)
axes[1].set(xlabel="receptors R", ylabel="MI [bits]",
            title="counting (points) against the KT bracket (band)")
for a in axes:
    a.grid(alpha=.2)
plt.tight_layout()
plt.show()
