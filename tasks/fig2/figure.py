# %%
"""fig2: does information keep rising as a cell expresses more genes?

Four arms crossing two binary choices. How gene sets are drawn, complete
coverage against bernoulli; and what a cell builds from them, every receptor
its genes allow against homomers only.

x is the REALISED mean genes per cell, not the target: a target of 2.25 is met
by mixing 2-gene and 3-gene cells, so the realised value is what was run.

The band is one standard deviation across independently drawn chemistries.
Each world draws its own latent_dim, n_ligands, family_spread and mixture
sparsity, so the band says how far a conclusion would move in a different
chemistry. A gap between arms no wider than the bands is not a result.

Everything plotted comes from grouped_counting, one estimator family, so
MI = H(Y) - H(Y|X) holds exactly rather than approximately.
"""
import matplotlib.pyplot as plt

from analysis.load import across_replicates, runs
from analysis.plot import band

GOAL = "fig2"
MI = "mutual_information_grouped_counting_mm"
H_Y = "response_entropy_grouped_counting_mm"
H_YX = "conditional_entropy_response_grouped_counting"
MISSING = "grouped_counting_missing_mass"

G, C = 5, 30              # which array the headline figure is for

frame = runs(GOAL, latest_only=True)
frame = frame.assign(arm=frame.strategy + "/" + frame.condition)

# Group by the TARGET, plot the REALISED mean. Under the bernoulli strategy the
# number of genes a cell gets is a Binomial draw, so the realised mean lands
# somewhere different in every world. Grouping by it would put every world on
# its own x with n = 1, and a point backed by one world has no band.
points = across_replicates(frame, keys=["arm", "n_genes", "n_cells", "target_genes"])

panel = points[(points.n_genes == G) & (points.n_cells == C)]
if panel.empty:
    raise SystemExit(f"No fig2 run at G={G}, C={C}. Run the four arm scripts first.")

# %%
fig, ax = plt.subplots(figsize=(7, 5))
band(ax, panel, "genes_per_cell", MI, by="arm")
ax.set(xlabel="genes expressed per cell (realised)", ylabel="MI [bits]",
       title=f"G = {G} genes available, C = {C} cells")
ax.legend()
plt.tight_layout()
plt.show()

# %%
# ── where the MI comes from: H(Y) minus the response noise H(Y|X) ───────────
# A curve that saturates because H(Y) stopped growing means something different
# from one that saturates because H(Y|X) caught up.
fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharex=True)
for ax, metric, label in ((axes[0], H_Y, "H(Y) [bits]"),
                          (axes[1], H_YX, "H(Y|X) [bits]")):
    band(ax, panel, "genes_per_cell", metric, by="arm")
    ax.set(xlabel="genes expressed per cell", ylabel=label)
axes[0].legend(fontsize=8)
plt.tight_layout()
plt.show()

# %%
# ── is the estimate trustworthy at every point? ─────────────────────────────
# Sampled counting underestimates H(Y) when the alphabet outruns the budget.
# Missing mass is the Good-Turing estimate of probability never sampled: where
# it rises, the MI above it is biased DOWN and the curve shape is suspect.
fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharex=True)
band(axes[0], panel, "genes_per_cell", MISSING, by="arm")
axes[0].set(xlabel="genes expressed per cell", ylabel="missing mass",
            title="probability never sampled")
for arm, curve in panel.groupby("arm"):
    axes[1].plot(curve.genes_per_cell, curve.n, "o-", label=arm)
axes[1].set(xlabel="genes expressed per cell", ylabel="independent worlds, n",
            title="how many worlds back each point")
axes[1].legend(fontsize=8)
plt.tight_layout()
plt.show()
