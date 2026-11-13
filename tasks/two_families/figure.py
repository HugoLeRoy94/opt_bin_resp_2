# %%
"""two_families: does the array split, and does the split match the prediction?

Two families seen with probability pi and 1 - pi, far enough apart that no
receptor serves both. If each group then behaves as an optimal single-family
array,

    MI_2 ~ H(F) + max_{R_1} [ pi G(R_1) + (1-pi) G(R-R_1) ] - H(F|A)

G(n) is the MI of an optimal SINGLE-family array of n receptors. Everything
drawn as a prediction is analytic: G is the closed form T.mi_eff at this run's
nu, and the split R_1* is the maximiser above. The closed form holds while each
family has at most D + 1 receptors. Past that it stays near-linear while the
measured single-family MI bends, so curves are drawn faint there.

Left: measured MI against the prediction, with H(F) marked as the floor the
family label alone would give. Right: the allocation. Receptors are assigned to
the family whose centre they sit nearest, read off the trained model, and
compared with the predicted R_1*, every receptor on the more probable family,
and a split proportional to pi.

One line per arm: the moved receptor refined before acceptance, or judged as
placed (train/moves.py).
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analysis.load import across_replicates, epochs, load_model, runs
from tasks._lib import theory as T
from tasks._lib import two_fam as TF

GOAL = "two_families"
MI = "codeword_entropy_mm"

frame = runs(GOAL, latest_only=True)
# family_weights is a config field, stored as "0.8,0.2"; pi is its first entry.
pi = float(str(frame.family_weights.iloc[0]).split(",")[0])

ARMS = {True: ("refined before acceptance", "tab:blue"),
        False: ("judged as placed", "tab:orange")}
points = across_replicates(frame, keys=["n_genes", "family_move_relax"]).sort_values("n_genes")
R_values = np.sort(frame.n_genes.unique())

# Analytic G(n), the closed form at this run's nu. G(0) = 0.
r0 = frame.iloc[0]
D = int(r0.latent_dim)
nu = T.nu_from_columns(D, r0.family_spread, r0.kernel_lambda, r0.fixed_max_energy,
                       r0.conc_std)
G_table = np.array([0.0] + [T.mi_eff(nu, n)["MI"] for n in range(1, R_values.max() + 1)])
G = lambda n: G_table[np.asarray(n, int)]
predicted = np.array([TF.h2(pi) + TF.best_split(G, int(R), pi)[1] for R in R_values])
split = np.array([TF.best_split(G, int(R), pi)[0] for R in R_values])
# Valid while no family holds more than D + 1 receptors.
split_valid = np.maximum(split, R_values - split) <= D + 1


def plot_valid(ax, x, y, valid, **kw):
    """Solid where the closed form holds, faint past it."""
    ax.plot(x[valid], y[valid], **kw)
    if (~valid).any():
        k = max(int(np.argmax(~valid)) - 1, 0)
        kw.pop("label", None)
        ax.plot(x[k:], y[k:], alpha=.3, **kw)

# %%
fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4))
for relax, (label, color) in ARMS.items():
    arm = points[points.family_move_relax == relax]
    axes[0].plot(arm.n_genes, arm[MI], "o-", color=color, label=label)
    axes[0].fill_between(arm.n_genes, arm[f"{MI} lo"], arm[f"{MI} hi"],
                         color=color, alpha=.15, linewidth=0)
plot_valid(axes[0], R_values, predicted, split_valid, ls="--", color="k", lw=1.2,
           label="analytic: H(F) + max split of G,  before subtracting H(F|A)")
plot_valid(axes[0], R_values, G(R_values), R_values <= D + 1, ls=":", color="0.5", lw=1,
           label="analytic: one family with all R receptors")
axes[0].axhline(TF.h2(pi), color="tab:red", lw=.8, ls=":",
                label=f"H(F) = {TF.h2(pi):.3f}, the family label alone")
axes[0].set(xscale="log", xlabel="receptors R", ylabel="MI [bits]",
            title=f"pi = {pi:g}, nu = {nu:.2f}, D = {D}")
axes[0].legend(fontsize=7)

# Allocation read off the trained model: a receptor belongs to the centre it
# sits nearest. This is geometry, independent of the MI it achieved.
observed = []
for _, row in frame.iterrows():
    env, _physics, receptor_indices, _readout = load_model(row.run_dir)
    array = TF.array_of(env, receptor_indices)
    group = np.argmin(((array["pos"][:, None, :] - array["centres"][None]) ** 2).sum(-1),
                      axis=1)
    observed.append(dict(n_genes=row.n_genes, family_move_relax=row.family_move_relax,
                         R_1=int((group == 0).sum())))
observed = across_replicates(pd.DataFrame(observed),
                             keys=["n_genes", "family_move_relax"]).sort_values("n_genes")

for relax, (label, color) in ARMS.items():
    arm = observed[observed.family_move_relax == relax]
    axes[1].plot(arm.n_genes, arm.R_1, "o-", color=color, label=label)
    axes[1].fill_between(arm.n_genes, arm["R_1 lo"], arm["R_1 hi"],
                         color=color, alpha=.15, linewidth=0)
plot_valid(axes[1], R_values, split, split_valid, ls="--", color="k", lw=1.2,
           label="analytic R_1*")
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
