"""Information-theory invariants the estimators must satisfy.

These are properties of the mathematics, not of any particular run, so they hold
whatever the physics does. Salvaged from the self-test blocks that used to live
at the bottom of bin_loss.py.

B = batch size, R = number of binary outputs. KT = Kolchinsky-Tracey.
"""
import math

import pytest
import torch

from estimate.entropy import (compute_kt_entropy, compute_kt_upper_entropy,
                              compute_shannon_joint_entropy)
from estimate.losses import ResponseEntropy


def soft(a):
    return torch.stack((1 - a, a), dim=-1)


def kt_reference(soft_assign, eps=1e-6):
    """Un-chunked KT lower bound: one dense (B, B) block, no tiling."""
    B = soft_assign.shape[0]
    A = soft_assign[:, :, 1].clamp(eps, 1.0 - eps)
    h_cond = (-(A * torch.log2(A) + (1.0 - A) * torch.log2(1.0 - A))).sum(dim=1).mean()
    sqrt_A, sqrt_1A = torch.sqrt(A), torch.sqrt(1.0 - A)
    bc = (sqrt_A.unsqueeze(1) * sqrt_A.unsqueeze(0)
          + sqrt_1A.unsqueeze(1) * sqrt_1A.unsqueeze(0))
    log_bc = torch.log(bc).sum(dim=2)
    inter = (torch.logsumexp(log_bc, dim=1) - math.log(B)).mean() / math.log(2)
    return h_cond - inter


def test_chunking_does_not_change_the_value():
    """Tiling is a memory strategy. It must not move the number."""
    torch.manual_seed(0)
    sa = soft(torch.rand(300, 17))
    assert compute_kt_entropy(sa, chunk_size=64).item() == pytest.approx(
        kt_reference(sa).item(), abs=1e-5)


@pytest.mark.parametrize("B,R", [(64, 12), (32, 10), (48, 8)])
def test_kt_brackets_the_exact_entropy(B, R):
    """KT lower <= exact H(s) <= KT upper, the whole point of the two bounds."""
    torch.manual_seed(B + R)
    sa = soft(torch.rand(B, R))
    lower = compute_kt_entropy(sa, chunk_size=16).item()
    exact = compute_shannon_joint_entropy(sa).item()
    upper = compute_kt_upper_entropy(sa, chunk_size=16).item()
    assert lower <= exact + 1e-4, f"lower bound violated: {lower} > {exact}"
    assert upper >= exact - 1e-4, f"upper bound violated: {upper} < {exact}"


def test_distinct_codewords_reach_log2_B():
    """B well-separated near-deterministic responses carry log2(B) bits."""
    torch.manual_seed(0)
    B, R = 32, 12
    codes = (torch.rand(B, R) > 0.5).float()
    A = codes * (1 - 1e-3) + (1 - codes) * 1e-3
    h_cond = (-(A * torch.log2(A) + (1 - A) * torch.log2(1 - A))).sum(1).mean().item()
    assert compute_kt_entropy(soft(A), chunk_size=8).item() == pytest.approx(
        math.log2(B) + h_cond, abs=0.01)


def test_identical_coin_flips_give_R_bits():
    """Every input giving A=0.5 means no separation: H = R, and zero MI."""
    R = 9
    A = torch.full((40, R), 0.5)
    assert compute_kt_entropy(soft(A), chunk_size=8).item() == pytest.approx(R, abs=1e-4)
    assert compute_kt_entropy(soft(A), chunk_size=8, return_mi=True).item() == pytest.approx(
        0.0, abs=1e-6)


def test_gradients_are_correct():
    """Finite-difference gradcheck in float64 through the soft assignment."""
    loss = ResponseEntropy(entropy_type="kt", collision_chunk_size=8)
    act = torch.rand(20, 6, dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(
        lambda a: compute_kt_entropy(loss.compute_soft_assignment(a), chunk_size=8),
        (act,), eps=1e-6, atol=1e-4)


def test_checkpointing_changes_memory_not_values():
    """recompute=True trades compute for memory. Values and grads must match."""
    torch.manual_seed(3)
    a = torch.rand(40, 8, requires_grad=True)
    b = a.detach().clone().requires_grad_(True)
    compute_kt_entropy(soft(a), chunk_size=8).backward()
    compute_kt_entropy(soft(b), chunk_size=8, recompute=True).backward()
    assert torch.allclose(a.grad, b.grad, atol=1e-6)
