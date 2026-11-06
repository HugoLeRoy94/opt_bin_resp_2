"""Receptor activation: the EC50-threshold limit of the MWC model.

MWC = Monod-Wyman-Changeux. EC50 = concentration at half activation. For a
heteromer it is the geometric mean of its subunit affinities, so ln EC50 is the
arithmetic mean of the subunit open-state energies.

    ln EC50_r = (1/k_sub) * sum_u E_open[u, l]            u over the k_sub subunits of r
    p_open_r  = sigmoid( logsumexp_l( ln c_l - ln EC50_rl ) / T )

k_sub = number of subunits per receptor (5 for a pentamer).
T = temperature of the threshold, the width of the sigmoid in log-concentration.
logsumexp over ligands aggregates mixture binding without overflow.
"""
import torch
import torch.nn as nn


class BinaryReceptor(nn.Module):
    def __init__(self, n_genes: int, k_sub: int, temperature: float):
        super().__init__()
        self.n_genes = n_genes
        self.k_sub = k_sub
        self.temperature = temperature

    def forward(self, energies, concentrations, receptor_indices,
                pre_gathered=False, composition=None):
        """Activation probabilities, shape (B, R).

        energies is one of
          (B, L, n_genes)        classic: one open-state energy per gene
          (B, L, R, k_sub)       interface model, pre_gathered=True
          (B, L, n_sources)      with `composition`: one energy per energy source
        concentrations is (B, L). receptor_indices is (R, k_sub).

        `composition` is an (n_sources, R) float matrix counting how many of
        receptor r's k_sub slots draw on source s. Given it, ln EC50 is one
        matmul and the (B, L, R, k_sub) intermediate is never built, which cuts
        peak memory by k_sub. See LigandEnvironment.bind_receptors.
        """
        if composition is not None:
            log_ec50 = torch.matmul(energies, composition) / self.k_sub
            c = concentrations.view(concentrations.shape[0], concentrations.shape[1], 1)
            return torch.clamp(self._p_open(c, log_ec50), 0.0, 1.0)

        if pre_gathered:
            energies_k = energies
        else:
            batch, n_ligands = energies.shape[0], energies.shape[1]
            gathered = energies[:, :, receptor_indices.view(-1)]
            energies_k = gathered.view(batch, n_ligands,
                                       receptor_indices.shape[0], self.k_sub)

        c = concentrations.view(concentrations.shape[0], concentrations.shape[1], 1)
        return torch.clamp(self._p_open(c, energies_k.mean(dim=-1)), 0.0, 1.0)

    def _p_open(self, c_reshaped, log_ec50):
        """Mixture aggregation then threshold, from ln EC50 of shape (B, L, R)."""
        ln_c = torch.log(c_reshaped + 1e-12)
        return torch.sigmoid(torch.logsumexp(ln_c - log_ec50, dim=1) / self.temperature)


@torch.no_grad()
def compute_initial_temperature(env, receptor_indices, calibration_batch_size=2048):
    """T_init = std of the pre-sigmoid term, so the sigmoid starts at unit spread.

    Without it the sigmoid starts either saturated (all 0 or 1, vanishing
    gradient) or mushy (all 0.5, non-discriminating).
    """
    composition = getattr(env, "composition", None)
    if composition is not None:
        E_open, concs, _ = env.sample_batch(calibration_batch_size,
                                            receptor_indices=receptor_indices)
        log_ec50 = torch.matmul(E_open, composition) / receptor_indices.shape[1]
    elif getattr(env, "use_interface_model", False):
        E_open, concs, _ = env.sample_batch(calibration_batch_size,
                                            receptor_indices=receptor_indices)
        log_ec50 = E_open.mean(dim=-1)
    else:
        E_open, concs, _ = env.sample_batch(calibration_batch_size)
        batch, n_ligands = E_open.shape[0], E_open.shape[1]
        gathered = E_open[:, :, receptor_indices.view(-1)]
        energies_k = gathered.view(batch, n_ligands, *receptor_indices.shape)
        log_ec50 = energies_k.mean(dim=-1)

    ln_c = torch.log(concs.unsqueeze(-1) + 1e-12)
    ln_sum_terms = torch.logsumexp(ln_c - log_ec50, dim=1)
    return max(ln_sum_terms.std().item(), 1e-3)
