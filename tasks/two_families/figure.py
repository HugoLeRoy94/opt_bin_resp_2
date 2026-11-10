# %%
"""two_families: does the array split, and does the split match the prediction?

Two families seen with probability pi and 1 - pi, far enough apart that no
receptor serves both. If each group then behaves as an optimal single-family
array,

    MI_2 ~ H(F) + max_{R_1} [ pi G(R_1) + (1-pi) G(R-R_1) ] - H(F|A)

G(n) is the MI of an optimal SINGLE-family array of n receptors, and it is not
measured here: it comes from the analytic_baseline sweep at D = 8,
conc_std = 0.2171, interpolated in log2 n. So analytic_baseline must be run and
measured first.

Left: measured MI against the prediction, with H(F) marked as the floor the
family label alone would give. Right: the allocation. Receptors are assigned to
the family whose centre they sit nearest, read off the trained model, and
compared with the predicted R_1*. The prediction is that while
pi dG(R_1) > (1-pi) G(1) every receptor goes to the more probable family, and
after that the weighted marginal gains equalise.

A split that matches the MI prediction while NOT matching the allocation would
mean the array found the same information a different way, which is a result
rather than a confirmation.
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analysis.load import across_replicates, load_model, runs
from tasks._lib import two_fam as TF

GOAL = "two_families"
G_GOAL = "analytic_baseline"    # where G(n) is measured
G_DIM, G_CONC_STD = 8, 0.2171   # the single-family condition this task copies
MI = "codeword_entropy_mm"

frame = runs(GOAL, latest_only=True)
# family_weights is a config field, stored as "0.8,0.2"; pi is its first entry.
pi = float(str(frame.family_weights.iloc[0]).split(",")[0])

# G(n) from analytic_baseline, at the matching single-family condition.
single = runs(G_GOAL, latest_only=True)
single = single[(single.latent_dim == G_DIM)
                & np.isclose(single.conc_std.astype(float), G_CONC_STD)]
if single.empty:
    raise SystemExit(
        f"No {G_GOAL} run at D={G_DIM}, conc_std={G_CONC_STD}. G(n) is measured "
        f"there, not here: run and measure {G_GOAL} first.")
g = across_replicates(single, keys=["n_genes"]).sort_values("n_genes")
G = TF.interp_G(g.n_genes.to_numpy(), g[MI].to_numpy())

points = across_replicates(frame, keys=["n_genes"]).sort_values("n_genes")
R_values = points.n_genes.to_numpy()
predicted = np.array([TF.h2(pi) + TF.best_split(G, int(R), pi)[1] for R in R_values])
split = np.array([TF.best_split(G, int(R), pi)[0] for R in R_values])

# %%
fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4))
axes[0].plot(R_values, points[MI], "o-", color="tab:blue", label="measured")
axes[0].fill_between(R_values, points[f"{MI} lo"], points[f"{MI} hi"],
                     color="tab:blue", alpha=.15, linewidth=0)
axes[0].plot(R_values, predicted, "--", color="k", lw=1.2,
             label="H(F) + max split of G,  before subtracting H(F|A)")
axes[0].plot(R_values, G(R_values), ":", color="0.5", lw=1,
             label="one family with all R receptors")
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
axes[1].plot(R_values, split, "--", color="k", lw=1.2, label="predicted R_1*")
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
