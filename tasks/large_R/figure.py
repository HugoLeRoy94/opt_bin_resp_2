# %%
"""large_R: measured information against the counting bounds, and the geometry.

Past R = D the array cannot keep its receptors mutually distinguishable, so the
information becomes a counting question: how many response patterns can R
hyperplanes cut out of a (D+1)-dimensional space?

    equal norms (shell)     ~ D log2 R        wastes one dimension
    unequal norms (ladder)  ~ (D+1) log2 R

Left panel: measured MI against both bounds. Right panel: the spread of the
learned receptor norms, which is what decides WHICH bound applies. A shell has
zero spread; a ladder has a lot. If the measured MI tracks the ladder bound,
the array should show spread norms, and if it tracks the shell it should not.

Observation noise is zero and the threshold near-hard, so H(Y|X) is about zero
and the hard-codeword entropy IS the mutual information.

The codeword estimate cannot exceed log2(B) however good the array is, so the
ceiling is drawn: a point sitting on it is reporting the measurement budget.
"""
import matplotlib.pyplot as plt
import numpy as np

from analysis.load import across_replicates, load_model, runs

GOAL = "large_R"
MI = "codeword_entropy_mm"
CEIL = "codeword_entropy_log2B"

frame = runs(GOAL, latest_only=True)
points = across_replicates(frame, keys=["latent_dim", "conc_std_log10", "n_genes"])
dims = sorted(points.latent_dim.unique())

# %%
fig, axes = plt.subplots(1, len(dims), figsize=(5.5 * len(dims), 4.4), squeeze=False)
for ax, D in zip(axes[0], dims):
    panel = points[points.latent_dim == D]
    stds = sorted(panel.conc_std_log10.unique())
    colours = plt.colormaps["viridis"](np.linspace(0, .85, len(stds)))
    for colour, cs in zip(colours, stds):
        curve = panel[panel.conc_std_log10 == cs].sort_values("n_genes")
        ax.plot(curve.n_genes, curve[MI], "o-", color=colour,
                label=f"conc_std = {cs:g}")
        ax.fill_between(curve.n_genes, curve[f"{MI} lo"], curve[f"{MI} hi"],
                        color=colour, alpha=.15, linewidth=0)
    one = panel.sort_values("n_genes").drop_duplicates("n_genes")
    ax.plot(one.n_genes, one.shell_bound, "--", color="k", lw=1,
            label=f"shell, ~{D} log2 R")
    ax.plot(one.n_genes, one.ladder_bound, ":", color="k", lw=1.2,
            label=f"ladder, ~{D + 1} log2 R")
    ax.plot(one.n_genes, one[CEIL], "-", color="0.7", lw=1, label="log2(B) ceiling")
    ax.set(xscale="log", xlabel="receptors R", ylabel="MI [bits]", title=f"D = {D}")
    ax.legend(fontsize=7)
    ax.grid(alpha=.2)
plt.tight_layout()
plt.show()

# %%
# ── which bound applies depends on whether the norms spread ─────────────────
# A shell is every receptor at one radius. A ladder spreads them. Reading the
# norms straight off the trained model says which the optimiser actually built,
# independently of the MI it achieved.
norms = []
for _, row in frame.iterrows():
    env, _physics, receptor_indices, _readout = load_model(row.run_dir)
    # Homomers: one receptor per gene, so a receptor's position IS its unit's.
    u = env.unit_latent.detach().cpu().numpy()[receptor_indices[:, 0].cpu().numpy()]
    radius = np.linalg.norm(u, axis=1)
    norms.append(dict(latent_dim=row.latent_dim, n_genes=row.n_genes,
                      conc_std_log10=row.conc_std_log10,
                      norm_spread=float(radius.std() / radius.mean())))
import pandas as pd                                                  # noqa: E402
spread = across_replicates(pd.DataFrame(norms),
                           keys=["latent_dim", "conc_std_log10", "n_genes"])

fig, axes = plt.subplots(1, len(dims), figsize=(5.5 * len(dims), 4), squeeze=False)
for ax, D in zip(axes[0], dims):
    panel = spread[spread.latent_dim == D]
    stds = sorted(panel.conc_std_log10.unique())
    colours = plt.colormaps["viridis"](np.linspace(0, .85, len(stds)))
    for colour, cs in zip(colours, stds):
        curve = panel[panel.conc_std_log10 == cs].sort_values("n_genes")
        ax.plot(curve.n_genes, curve.norm_spread, "o-", color=colour,
                label=f"conc_std = {cs:g}")
    ax.axhline(0, color="k", lw=.8, ls="--")
    ax.set(xscale="log", xlabel="receptors R",
           ylabel="norm spread, sd / mean", title=f"D = {D}: shell (0) or ladder (>0)")
    ax.legend(fontsize=7)
    ax.grid(alpha=.2)
plt.tight_layout()
plt.show()
