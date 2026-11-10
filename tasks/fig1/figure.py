# %%
"""fig1: information against gene count, one curve per heteromerization ratio.

Two uncertainties are shown and they mean different things.

  The BRACKET, solid against dashed, is the Kolchinsky-Tracey lower and upper
  bound on I(Y;X) for one trained array. It is a guarantee about the estimator,
  not noise, and more runs do not narrow it. The solid lower bound is what the
  optimizer maximised, so it is the number to read.

  The BAND around the solid line is one standard deviation across independently
  drawn chemical worlds. That is genuine variability, and it is what says
  whether a gap between two ratios is real.

A gap between two curves no wider than their bands is not a result.

Ratio 1 is the homomer condition: cascading fills complexity tiers in order and
tier 1 holds exactly n_genes rings, so R = n_genes is the homomer array. The
y = x line is the lossless-homomer reference, where every gene contributes one
fully independent bit.
"""
import matplotlib.pyplot as plt
import numpy as np

from analysis.load import across_replicates, runs
from analysis.plot import reference

GOAL = "fig1"
KT_LO = "mutual_information_kt"           # optimizer target, the reported MI
KT_UP = "mutual_information_kt_upper"     # top of the certified bracket
RATIOS = (1, 2, 3, 4, 5)

frame = runs(GOAL, latest_only=True)
frame = frame.assign(ratio=(frame.n_receptors / frame.n_genes).round().astype(int))
frame = frame[frame.ratio.isin(RATIOS)]
if frame.empty:
    raise SystemExit(f"No {GOAL} run with a ratio in {RATIOS}. "
                     f"Run tasks/fig1/sweep.py first.")

# One standard deviation across independently drawn worlds. Each world draws
# its own n_families, n_ligands, latent_dim, family_spread and family distance,
# so this band is the variability across a CATEGORY of chemistries.
points = across_replicates(frame, keys=["ratio", "n_genes"])

# %%
fig, ax = plt.subplots(figsize=(5.5, 4.2))
colours = plt.colormaps["viridis"]
norm = plt.Normalize(min(RATIOS), max(RATIOS))

for ratio in sorted(points.ratio.unique()):
    curve = points[points.ratio == ratio].sort_values("n_genes")
    if len(curve) < 2:
        continue
    colour = colours(norm(ratio))
    ax.plot(curve.n_genes, curve[KT_LO], "o-", color=colour,
            label=f"R/G = {ratio}" + ("  (homomers)" if ratio == 1 else ""))
    ax.fill_between(curve.n_genes, curve[f"{KT_LO} lo"], curve[f"{KT_LO} hi"],
                    color=colour, alpha=.15, linewidth=0)
    ax.plot(curve.n_genes, curve[KT_UP], "--", color=colour, linewidth=1, alpha=.7)

genes = np.sort(points.n_genes.unique())
ax.plot(genes, genes, ":", color="k", linewidth=1, alpha=.6,
        label="one bit per gene (y = x)")

ax.set(xlabel="genes available, G", ylabel="I(Y;X) [bits]",
       title="solid: KT lower bound   dashed: KT upper   band: 1 sd across worlds")
ax.spines[["top", "right"]].set_visible(False)
ax.legend(fontsize=8)
plt.tight_layout()
plt.show()

# %%
# ── how many worlds back each point, and how wide the bracket is ────────────
# A tight band on n = 1 is not a tight result, and a point whose bracket is
# wider than the effect under discussion cannot settle it.
fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
for ratio in sorted(points.ratio.unique()):
    curve = points[points.ratio == ratio].sort_values("n_genes")
    colour = colours(norm(ratio))
    axes[0].plot(curve.n_genes, curve.n, "o-", color=colour, label=f"R/G = {ratio}")
    axes[1].plot(curve.n_genes, curve[KT_UP] - curve[KT_LO], "o-", color=colour)
axes[0].set(xlabel="genes available, G", ylabel="independent worlds, n")
reference(axes[0], 1, label="n = 1: no band is meaningful")
axes[0].legend(fontsize=7)
axes[1].set(xlabel="genes available, G", ylabel="bracket width [bits]",
            title="KT upper minus KT lower")
for a in axes:
    a.spines[["top", "right"]].set_visible(False)
plt.tight_layout()
plt.show()
