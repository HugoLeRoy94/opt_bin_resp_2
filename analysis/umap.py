"""The chemical landscape panel: a 2-D view of the latent space.

UMAP = Uniform Manifold Approximation and Projection, a non-linear projection
that keeps neighbours together. It is used here purely to LOOK at a space with
more than two dimensions. Distances in the projection are not metric, so read
the panel for structure (what clusters with what, where the receptors sat down)
and never measure anything off it.

Two steps, kept separate so one fit can serve several panels:

    embedding = latent_embedding(env, receptor_indices)   # fits UMAP, slow
    draw_latent(ax, embedding)                            # draws, cheap

What is drawn: one coloured KDE cloud per ligand family, its exact centre, the
ligands belonging to it, and every receptor as a numbered marker placed at its
centroid in latent space. For the interface model a receptor's centroid is the
mean of its pocket midpoints, 0.5*(unit_latent_plus[i] + unit_latent_minus[j]),
which is where the pocket actually binds, not either face on its own.

umap-learn and seaborn are imported lazily and declared in the `analysis` extra:
a training container never needs them.
"""
import numpy as np
import torch


def latent_embedding(env, receptor_indices, n_samples_per_family=1000,
                     random_state=42):
    """Fit one UMAP over families, ligands, receptor centroids and a sampled cloud.

    All four are projected together so their positions are comparable. The cloud
    is drawn with env.sample_near_centers, the same draw the ligands themselves
    came from, so it cannot drift from the distribution the run actually used.
    """
    import umap                       # lazy: training never needs it

    if n_samples_per_family < 2:
        raise ValueError("n_samples_per_family must be at least 2")

    n_families, n_ligands = env.n_families, env.n_ligands
    n_receptors = receptor_indices.shape[0]

    v_families = env.family_latent.detach().cpu().numpy()
    v_ligands = env.ligand_latent.detach().cpu().numpy()
    assignments = env.ligand_family_assignments.detach().cpu().numpy()

    if env.use_interface_model:
        idx_i = receptor_indices
        idx_j = receptor_indices.roll(-1, dims=1)
        pockets = 0.5 * (env.unit_latent_plus[idx_i] + env.unit_latent_minus[idx_j])
        v_receptors = pockets.mean(dim=1).detach().cpu().numpy()
    else:
        v_receptors = env.unit_latent[receptor_indices].mean(dim=1).detach().cpu().numpy()

    labels = np.repeat(np.arange(n_families), n_samples_per_family)
    centers = env.family_latent[torch.as_tensor(labels, device=env.family_latent.device)]
    # fork_rng keeps this private seed out of the simulation's RNG stream, so
    # drawing a picture can never shift the numbers a later run produces.
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(random_state)
        cloud = env.sample_near_centers(centers).detach().cpu().numpy()

    stacked = np.vstack([v_families, v_ligands, v_receptors, cloud])
    reducer = umap.UMAP(n_neighbors=min(15, len(stacked) - 1), min_dist=0.1,
                        metric="euclidean", random_state=random_state, n_jobs=1)
    projected = reducer.fit_transform(stacked)

    a, b = n_families, n_families + n_ligands
    return dict(families=projected[:a], ligands=projected[a:b],
                receptors=projected[b:b + n_receptors],
                samples=projected[b + n_receptors:], sample_labels=labels,
                ligand_assignments=assignments, latent_dim=env.latent_dim)


def draw_latent(ax, embedding, show_legend=True, cloud_alpha=0.3):
    """Draw the landscape on `ax`. Returns `ax`. Creates nothing, saves nothing."""
    import matplotlib.pyplot as plt
    import seaborn as sns             # lazy: only the KDE clouds need it

    families = embedding["families"]
    ligands = embedding["ligands"]
    receptors = embedding["receptors"]
    samples = embedding["samples"]
    labels = embedding["sample_labels"]
    assignments = embedding["ligand_assignments"]
    n_families = len(families)

    colours = (plt.cm.tab10.colors if n_families <= 10
               else plt.cm.viridis(np.linspace(0, 1, n_families)))

    for f in range(n_families):
        colour = colours[f % len(colours)]
        # KDE contours are what make the family regions readable as regions
        # rather than a scatter of dots.
        sns.kdeplot(x=samples[labels == f, 0], y=samples[labels == f, 1], ax=ax,
                    fill=True, color=colour, alpha=cloud_alpha, levels=5, thresh=.05)
        ax.scatter(families[f, 0], families[f, 1], marker="o",
                   s=max(100 // n_families, 10), color=colour, edgecolor="black",
                   linewidth=1.2, zorder=4,
                   label=f"Fam {f}" if f < 10 else None)
        here = ligands[assignments == f]
        ax.scatter(here[:, 0], here[:, 1], marker=".", s=20, color=colour,
                   edgecolor="black", linewidth=.5, zorder=3)

    for r in range(len(receptors)):
        ax.text(receptors[r, 0], receptors[r, 1], str(r), fontsize=8,
                ha="center", va="center", fontweight="bold", zorder=5,
                bbox=dict(facecolor="white", edgecolor="black",
                          boxstyle="circle,pad=0.2", alpha=.8))

    ax.scatter([], [], marker="o", color="white", edgecolor="black",
               label="Receptors")                       # legend entry only
    ax.set(xlabel="UMAP Dimension 1", ylabel="UMAP Dimension 2",
           xticks=[], yticks=[])
    if show_legend:
        # bbox_to_anchor rather than shrinking the axes: resizing breaks the
        # layout when this panel is one cell of a larger figure.
        ax.legend(loc="center left", bbox_to_anchor=(1, .5), fontsize=8)
    return ax
