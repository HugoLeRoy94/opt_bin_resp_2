# %%
"""two_families: does the array split, and how are receptors allocated?

Two families seen with probability pi and 1 - pi, far enough apart that no
receptor serves both.

Left: measured MI against R, with H(F) marked as the floor the family label
alone would give. Right: the allocation. Receptors are assigned to the family
whose centre they sit nearest, read off the trained model, and compared with
the two naive allocations: every receptor on the more probable family, and a
split proportional to pi.
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analysis.load import across_replicates, epochs, load_model, runs
from tasks._lib import two_fam as TF

GOAL = "two_families"
MI = "codeword_entropy_mm"

frame = runs(GOAL, latest_only=True)
# family_weights is a config field, stored as "0.8,0.2"; pi is its first entry.
pi = float(str(frame.family_weights.iloc[0]).split(",")[0])

points = across_replicates(frame, keys=["n_genes"]).sort_values("n_genes")
R_values = points.n_genes.to_numpy()

# %%
fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4))
axes[0].plot(R_values, points[MI], "o-", color="tab:blue", label="measured")
axes[0].fill_between(R_values, points[f"{MI} lo"], points[f"{MI} hi"],
                     color="tab:blue", alpha=.15, linewidth=0)
axes[0].axhline(TF.h2(pi), color="tab:red", lw=.8, ls=":",
                label=f"H(F) = {TF.h2(pi):.3f}, the family label alone")
axes[0].set(xscale="log", xlabel="receptors R", ylabel="MI [bits]",
            title=f"pi = {pi:g}")
axes[0].legend(fontsize=7)

# Allocation read off the trained model: a receptor belongs to the centre it
# sits nearest. This is geometry, independent of the MI it achieved.
observed = []
for _, row in frame.iterrows():
    env, _physics, receptor_indices, _readout = load_model(row.run_dir)
    array = TF.array_of(env, receptor_indices)
    group = np.argmin(((array["pos"][:, None, :] - array["centres"][None]) ** 2).sum(-1),
                      axis=1)
    observed.append(dict(n_genes=row.n_genes, R_1=int((group == 0).sum())))
observed = across_replicates(pd.DataFrame(observed), keys=["n_genes"]).sort_values("n_genes")

axes[1].plot(observed.n_genes, observed.R_1, "o-", color="tab:blue", label="observed")
axes[1].fill_between(observed.n_genes, observed["R_1 lo"], observed["R_1 hi"],
                     color="tab:blue", alpha=.15, linewidth=0)
axes[1].plot(R_values, R_values, ":", color="0.5", lw=1, label="all on family 1")
axes[1].plot(R_values, pi * R_values, "-.", color="tab:green", lw=1,
             label=f"proportional, R_1 = {pi:g} R")
axes[1].set(xscale="log", xlabel="receptors R", ylabel="receptors on family 1",
            title="allocation, from the learned positions")
axes[1].legend(fontsize=7)
for ax in axes:
    ax.grid(alpha=.2)
plt.tight_layout()
plt.show()

# %%

# %%
# Convergence: the training KT MI after each epoch, one line per run, coloured
# by R. The dotted line is H(F): a curve that plateaus there has learned the
# family label and nothing within either family.
history = epochs(frame)
cmap = plt.get_cmap("viridis")
R_all = np.sort(frame.n_genes.unique())
fig, ax = plt.subplots(figsize=(6, 4))
for run_dir, part in history.groupby("run_dir"):
    R = part.n_genes.iloc[0]
    ax.plot(part.epoch, part.train_mutual_information, lw=.8,
            color=cmap(np.log2(R / R_all[0]) / max(np.log2(R_all[-1] / R_all[0]), 1)))
ax.axhline(TF.h2(pi), color="tab:red", lw=.8, ls=":")
sm = plt.cm.ScalarMappable(cmap=cmap,
                           norm=plt.matplotlib.colors.LogNorm(R_all[0], R_all[-1]))
fig.colorbar(sm, ax=ax, label="receptors R")
ax.set(xlabel="epoch", ylabel="training MI, KT bound [bits]",
       title=f"convergence, pi = {pi:g}")
ax.grid(alpha=.2)
plt.tight_layout()
plt.show()

# %%
