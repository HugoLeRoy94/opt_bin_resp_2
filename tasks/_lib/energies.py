"""Learned binding energies out of a trained environment, in natural-log units.

Both accessors go through the environment's own methods rather than reading a
parameter by name. Under shared_unit_energies there is a single scalar and no
per-gene attribute exists, so reading the parameter directly works in one mode
and raises in the other.

The simulator stores concentrations in log10 while the analytics is in natural
log. Anything combining these with conc_mean or conc_std must convert; see
theory.from_log10. Getting it wrong rescales nu by 2.3.
"""
import numpy as np
import torch.nn.functional as F


def emax(env) -> np.ndarray:
    """Learned selectivity ceiling per gene, shape (n_genes,)."""
    if hasattr(env, "max_energies_raw"):
        return F.softplus(env.max_energies_raw().detach()).cpu().numpy()
    return 0.5 * (F.softplus(env.max_energy_u_raw_plus.detach())
                  + F.softplus(env.max_energy_u_raw_minus.detach())).cpu().numpy()


def ebase(env) -> np.ndarray:
    """Learned base energy per gene, shape (n_genes,)."""
    if hasattr(env, "base_energies"):
        return env.base_energies().detach().cpu().numpy()
    return 0.5 * (env.base_energy_u_plus.detach()
                  + env.base_energy_u_minus.detach()).cpu().numpy()
