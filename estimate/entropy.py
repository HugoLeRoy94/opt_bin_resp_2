"""Entropy and mutual-information estimators for binary response arrays.

Notation: B = batch size, R = number of binary outputs (receptors or cells).
`soft_assign` is (B, R, 2); channel 1 is A[b,r] = P(output r active | input b),
channel 0 is 1 - A. The array's response is the uniform B-component mixture

    p(s) = (1/B) sum_b prod_r Bernoulli(A[b,r]),   s in {0,1}^R

and H(s) is its Shannon entropy in bits. H(Y|X) is the mean per-component
entropy, so I(Y;X) = H(s) - H(Y|X).

Three estimators of the same H(s):

  compute_shannon_joint_entropy   exact, enumerates all 2^R states.
                                  O(B * 2^R), usable only for R below about 15.
                                  The ground truth the others are checked against.

  compute_kt_entropy              Kolchinsky-Tracey LOWER bound, via the
                                  Bhattacharyya affinity between components.
                                  O(B^2 * R). Tight when components separate.

  compute_kt_upper_entropy        KT UPPER bound, same pairwise structure but
                                  the affinity is -KL instead of Bhattacharyya,
                                  which flips the bound direction. O(B^2 * R).

The two KT bounds bracket the true H(s). They are kept as separate functions on
purpose: they share only the chunked pairwise loop, while the kernel, the
auxiliary terms, the output clamp and the checkpointing all differ.

KT = Kolchinsky & Tracey, "Estimating Mixture Entropy with Pairwise Distances",
Entropy 2017.
"""
import torch
import math
from torch.utils.checkpoint import checkpoint

def compute_shannon_joint_entropy(soft_assign: torch.Tensor) -> torch.Tensor:
    """
    Computes the Shannon joint entropy of a discrete system from soft assignments
    using exact enumeration. Ideal for small state spaces.
    """
    # B : batch size
    # R : # of receptors
    # K : # of activity bins = 2
    B, R, K = soft_assign.shape

    # We iteratively build the exact (B, K^R) probability tensor
    joint_p = soft_assign[:, 0, :] # Start with receptor 0: (B, K)

    for r in range(1, R):
        # Multiply current combinations by the probabilities of the next receptor
        # (B, K^{r}, 1) * (B, 1, K) -> flat to (B, K^{r+1})
        joint_p = (joint_p.unsqueeze(-1) * soft_assign[:, r, :].unsqueeze(1)).view(B, -1)

    # Average across the batch to get the true probability of every possible state
    p_a = joint_p.mean(dim=0) # Shape: (K^R,)

    # Use clamp to prevent log2(0) while maintaining stable gradients
    p_a_safe = torch.clamp(p_a, min=1e-12)
    log_2_p = torch.log2(p_a_safe)
    joint_h = -torch.sum(p_a * log_2_p)

    return joint_h




def _kt_logbc_tile(sqrt_A_i, sqrt_1A_i, sqrt_A_j, sqrt_1A_j, multiplicities=None):
    """One (m,n) tile of log-Bhattacharyya: Σ_R log( √(A_iA_j) + √((1-A_i)(1-A_j)) ).

    Inputs (m,R) and (n,R); output (m,n). This is a pure elementwise+reduction chain
    (mul, mul, add, log, sum over R) — the ideal torch.compile target: Inductor fuses
    it into ~1 kernel AND can avoid materialising the (m,n,R) intermediate, which cuts
    both the kernel-launch count and the memory traffic that dominate KT (see the
    profiling task). Eager and compiled give identical values/gradients.
    """
    bc = (sqrt_A_i.unsqueeze(1) * sqrt_A_j.unsqueeze(0)
          + sqrt_1A_i.unsqueeze(1) * sqrt_1A_j.unsqueeze(0))   # (m, n, R)
    log_bc = torch.log(bc)
    if multiplicities is not None:
        log_bc = log_bc * multiplicities
    return log_bc.sum(dim=2)                                  # (m, n)


_kt_tile_compiled = None   # module-level cache for the torch.compile'd tile kernel


def _kt_tile_fn(use_compile: bool):
    """Return the tile kernel, lazily torch.compile-ing it once when requested."""
    global _kt_tile_compiled
    if not use_compile:
        return _kt_logbc_tile
    if _kt_tile_compiled is None:
        # dynamic=True: the trailing j/i chunk is smaller (n varies), so mark shapes
        # dynamic to avoid a fresh compile per edge-tile size.
        _kt_tile_compiled = torch.compile(_kt_logbc_tile, dynamic=True)
    return _kt_tile_compiled


def _kt_row_contribution(sqrt_A_i, sqrt_1A_i, sqrt_A, sqrt_1A, log_B, chunk_size,
                         tile_fn=_kt_logbc_tile, multiplicities=None):
    """Sum over the rows in one i-chunk of ( logsumexp_j logBC[i,:] - log_B ).

    Factored out of compute_kt_entropy so it can be gradient-checkpointed: its
    (m, n, R) Bhattacharyya blocks and (m, B) row are the memory bottleneck, and
    when wrapped in torch.utils.checkpoint they are recomputed in backward instead
    of retained — dropping the retained graph from O(B²·R) to O(B·R).
    sqrt_A_i / sqrt_1A_i are (m, R); sqrt_A / sqrt_1A are the full (B, R).
    tile_fn is _kt_logbc_tile (eager) or its torch.compile'd version.
    """
    B = sqrt_A.shape[0]
    row_blocks = []
    for j0 in range(0, B, chunk_size):
        j1 = min(j0 + chunk_size, B)
        row_blocks.append(tile_fn(sqrt_A_i, sqrt_1A_i, sqrt_A[j0:j1], sqrt_1A[j0:j1],
                                  multiplicities))  # (m, n)
    full_row = torch.cat(row_blocks, dim=1)                            # (m, B)
    return (torch.logsumexp(full_row, dim=1) - log_B).sum()


KT_EPS = 1e-6


def compute_response_conditional_entropy(activity: torch.Tensor,
                                         eps: float = KT_EPS,
                                         multiplicities=None) -> torch.Tensor:
    """Mean H(Y|X) for conditionally independent Bernoulli outputs, in bits.

    X includes the full sampled input, including any observation-noise realization.
    KT and stochastic counting evaluation use this same probability clamp.
    """
    a = activity.clamp(eps, 1.0 - eps)
    entropy = -(a * torch.log2(a) + (1.0 - a) * torch.log2(1.0 - a))
    if multiplicities is not None:
        entropy = entropy * multiplicities
    return entropy.sum(-1).mean()


def compute_kt_entropy(
    soft_assign: torch.Tensor,
    chunk_size: int = 512,
    eps: float = KT_EPS,
    recompute: bool = False,
        use_compile: bool = False,
    return_mi: bool = False,
    multiplicities=None,
) -> torch.Tensor:
    """Kolchinsky-Tracey Bhattacharyya LOWER bound on the joint Shannon entropy.

    Certified lower bound on the SAME H(s) that compute_shannon_joint_entropy
    computes exactly: the entropy of the uniform B-component mixture
    p(s) = (1/B) Σ_b Π_r Bernoulli(A_br), s ∈ {0,1}^R, A_br = soft_assign[b,r,1].
    Tight in the well-separated / noiseless regime.
    Ref: Kolchinsky & Tracey, "Estimating Mixture Entropy with Pairwise
    Distances", Entropy 2017 (Bhattacharyya-affinity pairwise bound).

    Formula (bits):
      h2(a)  = -a·log2(a) - (1-a)·log2(1-a)            binary Shannon entropy
      H_cond = mean_b Σ_r h2(A_br)                     mean component entropy
      logBC[i,j] = Σ_r log( √(A_i A_j) + √((1-A_i)(1-A_j)) )   (natural log)
      H_lb   = H_cond - (1/B) Σ_i ( logsumexp_j logBC[i,:] - log B ) / log 2

    BC[i,j] is the Bhattacharyya affinity between components i and j;
    BC[i,i]=1, so the diagonal anchors the inner sum at 1/B and the
    separation-term ceiling is log2(TOTAL batch B) — NOT log2(chunk).
    The entropy bound itself can reach H_cond + log2(B).

    ALL-PAIRS, DIAGONAL INCLUDED, NORMALIZED BY TOTAL B. The inner j-loop
    spans the ENTIRE batch and the diagonal (logBC=0) is kept. This is NOT
    the adjacent-chunk / diagonal-masked scheme of the collision estimator.
    Pairwise arithmetic is O(B²·R), independent of tile size. Inference also
    retains O(chunk·B) row buffers alongside the O(chunk²·R) tile.

    Args:
        soft_assign: (B, R, 2) tensor; channel 1 is A_br, channel 0 is 1-A_br.
        chunk_size:  edge length m of the (m, n) logBC blocks.
        eps:         clamp A into [eps, 1-eps] for log/sqrt stability.
        return_mi:   return the separation term alone, bounding I(Y;X) for the
                     empirical mixture. Avoids subtracting two large entropy terms.
        multiplicities: optional positive cell counts for structurally identical
                     Bernoulli columns. Weights log-BC and conditional entropy;
                     entropy still refers to all labeled binary outputs.
        recompute:   if True, gradient-checkpoint each i-chunk's contribution so its
                     (m, n, R) / (m, B) tensors are recomputed in backward instead of
                     retained. Retained graph drops O(B²·R) → O(B·R), so the training
                     batch is bounded by compute, not memory (see resolve_batch_sizes,
                     KT_COMPUTE_BUDGET). Exact — no effect on value or gradients.
                     Costs ~one extra forward of the inner loop in backward.

    Returns:
        Scalar lower bound in bits, differentiable w.r.t. soft_assign.
    """
    B, R, _ = soft_assign.shape
    A = soft_assign[:, :, 1].clamp(eps, 1.0 - eps)   # (B, R)

    # Conditional (mean per-component) entropy in bits.
    # Repeated identical Bernoulli factors can be represented by one column with
    # its multiplicity. Binomial BC equals the single-cell BC to this power.
    h_cond = None if return_mi else compute_response_conditional_entropy(A, eps, multiplicities)

    sqrt_A  = torch.sqrt(A)                            # (B, R)
    sqrt_1A = torch.sqrt(1.0 - A)                      # (B, R)
    log_B = math.log(B)

    # Σ_i ( logsumexp_j logBC[i,:] - log B ), all pairs, diagonal kept.
    # Per i-chunk contribution is optionally gradient-checkpointed (recompute) and the
    # per-tile kernel is optionally torch.compile'd (use_compile) to fuse launches.
    do_ckpt = recompute and soft_assign.requires_grad
    tile_fn = _kt_tile_fn(use_compile)
    inter_nats = soft_assign.new_zeros(())
    for i0 in range(0, B, chunk_size):
        i1 = min(i0 + chunk_size, B)
        sa_i, s1a_i = sqrt_A[i0:i1], sqrt_1A[i0:i1]   # (m, R)
        if do_ckpt:
            contrib = checkpoint(_kt_row_contribution, sa_i, s1a_i, sqrt_A, sqrt_1A,
                                 log_B, chunk_size, tile_fn, multiplicities, use_reentrant=False)
        else:
            contrib = _kt_row_contribution(sa_i, s1a_i, sqrt_A, sqrt_1A, log_B, chunk_size,
                                           tile_fn, multiplicities)
        inter_nats = inter_nats + contrib

    inter_bits = (inter_nats / B) / math.log(2)
    return -inter_bits if return_mi else h_cond - inter_bits


def compute_kt_upper_entropy(
    soft_assign: torch.Tensor,
    chunk_size: int = 512,
    eps: float = KT_EPS,
    return_mi: bool = False,
    multiplicities=None,
) -> torch.Tensor:
    """Kolchinsky-Tracey UPPER bound on the joint Shannon entropy H(s).

    Companion to compute_kt_entropy (the Bhattacharyya LOWER bound): same
    pairwise-distance structure, but the component affinity uses the KL divergence
    instead of the Bhattacharyya coefficient, which flips the bound direction.

        H(s) <= H_cond - (1/B) Sum_i log( (1/B) Sum_j exp(-KL(p_i || p_j)) )

    For product-Bernoulli components (A_ir = P(receptor r active | sample i)):
        KL(i||j) = Sum_r [ A_ir log(A_ir/A_jr) + (1-A_ir) log((1-A_ir)/(1-A_jr)) ]
                 = ent_i - cross(i,j)
    with the i-only self term  ent_i = Sum_r [A_ir log A_ir + (1-A_ir) log(1-A_ir)]
    and the cross term  cross(i,j) = Sum_r [A_ir log A_jr + (1-A_ir) log(1-A_jr)].
    The affinity kernel is -KL(i||j) = cross(i,j) - ent_i.

    Certified upper bound (never below the true H(s)); tight in the well-separated
    regime. The diagonal (i=j, KL=0) is kept and the sum normalised by the total
    batch B, so the MI separation term is capped at log2(B), as for the lower bound;
    the entropy includes H_cond in addition.
    Pairwise arithmetic O(B²·R); inference memory O(chunk²·R + chunk·B).

    Args:
        soft_assign: (B, R, 2); channel 1 is A_ir, channel 0 is 1-A_ir.
        chunk_size:  edge length m of the (m, n) blocks.
        eps:         clamp A into [eps, 1-eps] for log stability.
        return_mi:   return the upper bound on I(Y;X), including the C-H(Y|X) cap.
        multiplicities: optional counts of identical cells represented by each
                     column. Weights KL and entropy; the binary cap is sum(counts).
    """
    B, R, _ = soft_assign.shape
    A = soft_assign[:, :, 1].clamp(eps, 1.0 - eps)            # (B, R)

    h_cond = compute_response_conditional_entropy(A, eps, multiplicities)
    weights = A.new_ones(R) if multiplicities is None else multiplicities.to(A)
    n_outputs = weights.sum()

    log_A  = torch.log(A)                                     # (B, R)
    log_1A = torch.log1p(-A)                                  # (B, R) = log(1-A)
    ent    = ((A * log_A + (1.0 - A) * log_1A) * weights).sum(dim=1)
    log_B  = math.log(B)

    inter_nats = soft_assign.new_zeros(())
    for i0 in range(0, B, chunk_size):
        i1 = min(i0 + chunk_size, B)
        A_i     = A[i0:i1].unsqueeze(1)                       # (m, 1, R)
        one_A_i = (1.0 - A[i0:i1]).unsqueeze(1)               # (m, 1, R)
        ent_i   = ent[i0:i1].unsqueeze(1)                     # (m, 1)
        row_blocks = []
        for j0 in range(0, B, chunk_size):
            j1 = min(j0 + chunk_size, B)
            log_A_j  = log_A[j0:j1].unsqueeze(0)              # (1, n, R)
            log_1A_j = log_1A[j0:j1].unsqueeze(0)             # (1, n, R)
            cross = ((A_i * log_A_j + one_A_i * log_1A_j) * weights).sum(dim=2)
            row_blocks.append(cross - ent_i)                 # (m, n) = -KL(i||j)
        full_row = torch.cat(row_blocks, dim=1)              # (m, B)
        inter_nats = inter_nats + (torch.logsumexp(full_row, dim=1) - log_B).sum()

    inter_bits = (inter_nats / B) / math.log(2)
    # H(s) of R binary receptors is <= R bits, so clamp: min of two valid upper
    # bounds is still an upper bound, and tighter in the overlapping regime.
    if return_mi:
        return torch.minimum(-inter_bits, n_outputs - h_cond)
    return torch.minimum(h_cond - inter_bits, n_outputs)
