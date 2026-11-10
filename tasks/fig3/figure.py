# %%
"""fig3: does the MI maximum move with k_sub?

Left: MI against genes per cell, one curve per k_sub, with the peak marked.
Right: the receptor pool on the same axis, which is the thing k_sub actually
changes.

The prediction from the pool-versus-dilution account is that the peak moves
LEFT as k_sub falls, to g=1 at k=2 and g=2 at k=3 and k=5. Lowering k_sub does
slow dilution, but it slows the pool by at least as much, and only the ratio
matters.

At k=2 the pool is exhausted at g=2, so that arm is a pure-dilution control:
any decline past g=2 has zero pool gain behind it.

The three arms share their gene sets, so k_sub is the only difference. The band
is one standard deviation across independently drawn chemistries.
"""
import matplotlib.pyplot as plt

from analysis.load import across_replicates, runs
from analysis.plot import band

GOAL = "fig3"
MI = "mutual_information_grouped_counting_mm"
KT_LO = "mutual_information_kt"
KT_UP = "mutual_information_kt_upper"
MISSING = "grouped_counting_missing_mass"
STYLE = {2: ("o-", "tab:blue"), 3: ("s-", "tab:green"), 5: ("^-", "tab:red")}

frame = runs(GOAL, latest_only=True)
points = across_replicates(frame, keys=["k_sub", "genes_per_cell"])

# %%
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
for k_sub in sorted(points.k_sub.unique()):
    curve = points[points.k_sub == k_sub].sort_values("genes_per_cell")
    marker, colour = STYLE.get(int(k_sub), ("o-", None))
    axes[0].plot(curve.genes_per_cell, curve[MI], marker, color=colour,
                 label=f"k_sub={int(k_sub)}")
    axes[0].fill_between(curve.genes_per_cell, curve[f"{MI} lo"], curve[f"{MI} hi"],
                         color=colour, alpha=.15, linewidth=0)
    peak = curve.loc[curve[MI].idxmax()]
    axes[0].axvline(peak.genes_per_cell, color=colour, lw=.8, ls=":")
    axes[1].plot(curve.genes_per_cell, curve.n_receptors_resolved, marker,
                 color=colour, label=f"k_sub={int(k_sub)}")
axes[0].set(xlabel="genes expressed per cell", ylabel="MI [bits]",
            title="dotted line = peak position")
axes[1].set(xlabel="genes expressed per cell", ylabel="receptors in the pool",
            yscale="log", title="what k_sub actually changes")
for ax in axes:
    ax.grid(alpha=.2)
    ax.legend(fontsize=8)
fig.suptitle(f"G = {int(points.n_genes.iloc[0])}, C = {int(points.n_cells.iloc[0])}, "
             f"complete coverage, heteromers")
plt.tight_layout()
plt.show()

# %%
# ── the cross-check: counting against the KT bracket ────────────────────────
# The two estimators fail in opposite directions, counting as the cell count
# rises and KT as the batch rises. Where counting leaves the bracket, at least
# one of them is out of its range and the point should not be read.
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
for k_sub in sorted(points.k_sub.unique()):
    curve = points[points.k_sub == k_sub].sort_values("genes_per_cell")
    marker, colour = STYLE.get(int(k_sub), ("o-", None))
    axes[0].plot(curve.genes_per_cell, curve[MI], marker, color=colour,
                 label=f"k_sub={int(k_sub)} counting")
    axes[0].fill_between(curve.genes_per_cell, curve[KT_LO], curve[KT_UP],
                         color=colour, alpha=.12, linewidth=0)
band(axes[1], points, "genes_per_cell", MISSING, by="k_sub")
axes[0].set(xlabel="genes expressed per cell", ylabel="MI [bits]",
            title="counting MI, band = KT bracket")
axes[1].set(xlabel="genes expressed per cell", ylabel="missing mass",
            title="probability never sampled by counting")
for ax in axes:
    ax.grid(alpha=.2)
    ax.legend(fontsize=8)
plt.tight_layout()
plt.show()
