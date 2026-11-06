"""Training objectives. All of them maximize a bound on I(Y;X).

I(Y;X) = H(Y) - H(Y|X), where Y is the array response and X the stimulus.

  DiscreteExactLoss             maximizes H(Y) alone, with entropy_type picking
                                the estimator ('shannon' exact, 'kt' lower bound).
                                Correct only when the readout is noiseless, since
                                it never subtracts H(Y|X).
  KTMutualInformationLoss       maximizes the KT lower bound on I(Y;X) directly,
                                without forming the difference of two large
                                entropies. This is the objective to prefer.
  GroupedCellMutualInformationLoss
                                exact I(Y;X) over grouped cell counts, by
                                enumerating joint count states. The reference the
                                KT version is validated against; cost grows with
                                prod_j (n_j + 1), hence cell_grouped_max_states.
  GroupedKTMutualInformationLoss
                                the KT bound on grouped counts. Same convention,
                                no joint-state enumeration, so it scales.
"""
import math

import torch
import torch.nn as nn

from estimate.entropy import (compute_kt_entropy, compute_kt_upper_entropy,
                              compute_shannon_joint_entropy)
from estimate.grouping import CellGrouping


class DiscreteExactLoss(nn.Module):
    """Maximize the joint entropy H(Y) of the binary response array.

    entropy_type selects the estimator used for training and for compute_entropy:
      'shannon'  exact enumeration, O(B * 2^R), small R only
      'kt'       Kolchinsky-Tracey lower bound, O(B^2 * R), scalable

    Train with one and measure with another by passing entropy_type to
    compute_entropy.
    """

    _ENTROPY_FNS = ('shannon', 'kt')

    def __init__(self, entropy_type: str, collision_chunk_size: int = 2048,
                 recompute_backward: bool = False, compile_kt: bool = False):
        super().__init__()
        self.entropy_type = entropy_type
        self.collision_chunk_size = collision_chunk_size
        self.recompute_backward = recompute_backward
        self.compile_kt = compile_kt

    def compute_soft_assignment(self, activity: torch.Tensor) -> torch.Tensor:
        return torch.stack([1.0 - activity, activity], dim=-1)

    def compute_entropy(self, activity: torch.Tensor,
                        entropy_type: str = None) -> torch.Tensor:
        """Joint entropy in bits, a positive scalar."""
        etype = entropy_type if entropy_type is not None else self.entropy_type
        soft_assign = self.compute_soft_assignment(activity)
        if etype == 'shannon':
            return compute_shannon_joint_entropy(soft_assign)
        return compute_kt_entropy(soft_assign, chunk_size=self.collision_chunk_size,
                                  recompute=self.recompute_backward,
                                  use_compile=self.compile_kt)

    def forward(self, activity: torch.Tensor) -> torch.Tensor:
        return -self.compute_entropy(activity)


class KTMutualInformationLoss(DiscreteExactLoss):
    """Maximize the KT lower bound on I(Y;X).

    compute_entropy still returns H(Y); forward returns -I_lower and
    differentiates through the whole objective. No output bits are sampled
    during training.
    """

    def __init__(self, **kwargs):
        super().__init__(entropy_type='kt', **kwargs)

    def forward(self, activity: torch.Tensor) -> torch.Tensor:
        return -compute_kt_entropy(
            self.compute_soft_assignment(activity),
            chunk_size=self.collision_chunk_size,
            recompute=self.recompute_backward,
            use_compile=self.compile_kt,
            return_mi=True,
        )


class GroupedCellMutualInformationLoss(DiscreteExactLoss):
    """Cell-only count enumeration; ``compute_entropy`` still returns full H(Y).

    Inputs remain (batch, cells), with one probability per labeled cell. Identical
    W rows must use the same readout parameters and conditionally independent
    output noise. The runner guarantees this for its shared cell readout.
    """

    def __init__(self, W: torch.Tensor, max_states: int = 65536):
        super().__init__(entropy_type="shannon")
        self.entropy_type = "grouped_mi"
        if max_states < 1:
            raise ValueError("cell_grouped_max_states must be positive.")
        self.grouping = W if isinstance(W, CellGrouping) else CellGrouping(W)
        self.n_cells = self.grouping.n_cells
        self.n_groups = self.grouping.n_groups
        self.n_states = self.grouping.n_states
        sizes = self.grouping.group_sizes
        if self.n_states > max_states:
            raise ValueError(
                f"Grouped cell alphabet has {self.n_states} states, exceeding "
                f"cell_grouped_max_states={max_states}. Select grouped_kt_mi with "
                "grouped_counting instead of exact grouped_information, or raise "
                "the limit after checking memory.")
        radices = sizes + 1
        strides = torch.cat((radices.new_ones(1), radices.cumprod(0)[:-1]))
        states = (torch.arange(self.n_states, device=sizes.device)[:, None]
                  // strides[None, :]) % radices[None, :]
        remaining = sizes[None, :] - states
        # Cache coefficients in double precision; cast to the activity dtype at
        # evaluation. A joint count has prod_j choose(n_j, k_j) labeled patterns.
        log_multiplicity = self.grouping.log_choose(sizes[None, :], states).sum(1)
        self.register_buffer("count_states", states)
        self.register_buffer("remaining_counts", remaining)
        self.register_buffer("log_multiplicity", log_multiplicity)

    def sufficient_statistics(self, activity: torch.Tensor):
        """Return sums of P(K|x) and H(K|x), mergeable across input chunks.

        No binary draws, no B² comparisons, and no (B,S,J) broadcast. The main
        work is two (B,J) @ (J,S) products, and storage is O(BS + SJ + BC).
        """
        p = self.grouping.probabilities(activity)
        log_joint = (p.log() @ self.count_states.to(p.dtype).T
                     + torch.log1p(-p) @ self.remaining_counts.to(p.dtype).T
                     + self.log_multiplicity.to(p.dtype)[None, :])
        # The exact rows sum to one; normalization suppresses floating-point
        # drift in large binomial coefficients without changing the channel.
        log_joint = log_joint - torch.logsumexp(log_joint, dim=1, keepdim=True)
        joint = log_joint.exp()
        conditional_sum = -(joint * log_joint).sum() / math.log(2)
        return joint.sum(0), conditional_sum

    def metrics_from_statistics(self, probability_sum, conditional_sum, n_samples):
        """Restore both entropies of labeled outputs; only count keys mean H(K)."""
        marginal = probability_sum / n_samples
        h_count = -(marginal * marginal.clamp_min(
            torch.finfo(marginal.dtype).tiny).log2()).sum()
        h_count_cond = conditional_sum / n_samples
        multiplicity = (marginal * self.log_multiplicity.to(marginal.dtype)).sum() / math.log(2)
        return {
            "grouped_count_entropy": h_count,
            "grouped_count_conditional_entropy": h_count_cond,
            "grouped_label_entropy": multiplicity,
            "response_entropy_grouped": h_count + multiplicity,
            "conditional_entropy_response_grouped": h_count_cond + multiplicity,
            "mutual_information_grouped": h_count - h_count_cond,
        }

    def compute_metrics(self, activity):
        return self.metrics_from_statistics(
            *self.sufficient_statistics(activity), activity.shape[0])

    def compute_entropy(self, activity, entropy_type=None):
        if entropy_type in (None, "grouped_mi", "shannon"):
            return self.compute_metrics(activity)["response_entropy_grouped"]
        return super().compute_entropy(activity, entropy_type)

    def forward(self, activity):
        return -self.compute_metrics(activity)["mutual_information_grouped"]


class GroupedKTMutualInformationLoss(KTMutualInformationLoss):
    """Same KT bound and labeled-entropy convention, with weighted group columns."""
    def __init__(self, W, **kwargs):
        super().__init__(**kwargs)
        self.grouping = W if isinstance(W, CellGrouping) else CellGrouping(W)

    def bound(self, activity, *, upper=False, return_mi=False):
        p = self.grouping.probabilities(activity)
        return self.bound_from_probabilities(p, upper=upper, return_mi=return_mi)

    def bound_from_probabilities(self, p, *, upper=False, return_mi=False):
        soft = torch.stack((1 - p, p), -1)
        kwargs = dict(chunk_size=self.collision_chunk_size, return_mi=return_mi,
                      multiplicities=self.grouping.group_sizes)
        if upper:
            return compute_kt_upper_entropy(soft, **kwargs)
        return compute_kt_entropy(soft, recompute=self.recompute_backward,
                                  use_compile=self.compile_kt, **kwargs)

    def compute_entropy(self, activity, entropy_type=None):
        if entropy_type in (None, 'kt', 'grouped_kt_mi'):
            return self.bound(activity)
        return super().compute_entropy(activity, entropy_type)

    def forward(self, activity):
        return -self.bound(activity, return_mi=True)
