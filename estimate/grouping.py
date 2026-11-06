"""Identical cells grouped into one column, and sampled counting over groups.

Cells with identical abundance rows W are structurally interchangeable: they see
the same drive and differ only by independent output noise. Grouping them turns
the response from "which cells fired" into "how many fired in each group", which
shrinks the alphabet from 2^C to prod_j (n_j + 1) without losing information,
because the labelled entropy is recovered by adding back the multiplicity term
log2 prod_j choose(n_j, k_j).

Groups are fixed by identical W rows. They are never inferred from similar
sampled activities, which would make the grouping depend on the draw.

n_j = size of group j. C = sum_j n_j = number of cells. J = number of groups.
"""
import math

import torch
import torch.nn as nn

from estimate.counting import SymbolCounter
from estimate.entropy import KT_EPS, compute_response_conditional_entropy

class CellGrouping(nn.Module):
    """Grouping metadata and per-group operations; never builds joint states."""

    def __init__(self, W):
        super().__init__()
        if W.ndim != 2 or min(W.shape) == 0:
            raise ValueError("Grouping requires a nonempty cell abundance matrix W.")
        _, inverse, sizes = torch.unique(W.detach(), dim=0, return_inverse=True,
                                        return_counts=True)
        self.n_cells = W.shape[0]
        self.n_groups = sizes.numel()
        self.n_states = math.prod((sizes + 1).tolist())  # Python int: no overflow
        self.register_buffer("cell_to_group", inverse)
        self.register_buffer("group_sizes", sizes)
        # Flat individual binomial supports: C+J entries, not product(n_j+1).
        ids = torch.arange(self.n_groups, device=W.device).repeat_interleave(sizes + 1)
        starts = (sizes + 1).cumsum(0) - (sizes + 1)
        k = torch.arange(self.n_cells + self.n_groups, device=W.device) - starts[ids]
        self.register_buffer("support_groups", ids)
        self.register_buffer("support_counts", k)
        self.register_buffer("support_log_choose", self.log_choose(sizes[ids], k))

    @staticmethod
    def log_choose(n, k):
        return (torch.lgamma(n.double() + 1) - torch.lgamma(k.double() + 1)
                - torch.lgamma((n - k).double() + 1))

    def probabilities(self, activity):
        if activity.ndim != 2 or activity.shape[1] != self.n_cells or not activity.shape[0]:
            raise ValueError(f"Expected nonempty (batch, {self.n_cells}) cell activity.")
        a = activity if activity.dtype == torch.float64 else activity.float()
        a = a.clamp(KT_EPS, 1 - KT_EPS)
        # Average duplicates so partial gradients agree with the ungrouped loss.
        total = a.new_zeros((a.shape[0], self.n_groups)).scatter_add(
            1, self.cell_to_group[None, :].expand(a.shape[0], -1), a)
        return (total / self.group_sizes).clamp(KT_EPS, 1 - KT_EPS)

    def conditional_entropies(self, p):
        """Mean count H(K|X), labeled H(Y|X), and their difference D, in bits."""
        ids, k = self.support_groups, self.support_counts
        log_pmf = (self.support_log_choose.to(p.dtype)
                   + p[:, ids].log() * k
                   + torch.log1p(-p[:, ids]) * (self.group_sizes[ids] - k))
        h_count = -(log_pmf.exp() * log_pmf).sum(1).mean() / math.log(2)
        h_response = compute_response_conditional_entropy(p, multiplicities=self.group_sizes)
        return h_count, h_response, h_response - h_count

    def sample(self, p, generator=None):
        return torch.binomial(self.group_sizes.to(p.dtype).expand_as(p), p,
                              generator=generator).to(torch.int64)

    def diagnostics(self):
        return dict(grouped_n_groups=self.n_groups, grouped_n_states=self.n_states,
                    grouped_count_entropy_upper=math.log2(self.n_states))


class GroupedResponseCounter:
    """Stream binomial draws and analytical noise; never enumerate joint states."""
    def __init__(self, grouping):
        self.grouping = grouping
        # Group counts are bounded by n_j, so a response packs into one int64
        # whenever prod(n_j+1) allows. SymbolCounter falls back to row storage.
        self.counter = SymbolCounter((grouping.group_sizes + 1).tolist())
        self.conditional_sum = 0.0
        self.response_conditional_sum = 0.0

    @torch.no_grad()
    def update(self, activity, generator=None):
        p = self.grouping.probabilities(activity)
        h_count, h_response, _ = self.grouping.conditional_entropies(p)
        n = activity.shape[0]
        self.conditional_sum += n * h_count.item()
        self.response_conditional_sum += n * h_response.item()
        self.counter.update(self.grouping.sample(p, generator))

    def metrics(self):
        plugin, mm, unique, log_n, missing = self.counter.entropy()
        n = self.counter.n_samples
        hc = self.conditional_sum / n
        hyc = self.response_conditional_sum / n
        d = hyc - hc
        return dict(
            grouped_count_entropy_counting_plugin=plugin,
            grouped_count_entropy_counting_mm=mm,
            grouped_count_conditional_entropy_counting=hc,
            grouped_label_entropy_counting=d,
            response_entropy_grouped_counting_plugin=plugin + d,
            response_entropy_grouped_counting_mm=mm + d,
            conditional_entropy_response_grouped_counting=hyc,
            mutual_information_grouped_counting_plugin=plugin - hc,
            mutual_information_grouped_counting_mm=mm - hc,
            grouped_counting_K_hat=unique,
            grouped_counting_unique_fraction=unique / n,
            grouped_counting_missing_mass=missing,
            grouped_counting_samples=n,
            grouped_counting_log2B=log_n,
            **self.grouping.diagnostics())
