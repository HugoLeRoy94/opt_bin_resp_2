"""Chemical latent space, ligand sampling, and open-state energies.

LigandEnvironment holds the learnable parameters (unit positions in the latent
space, binding energies) and the fixed buffers (family centres, ligand
coordinates). sample_batch() produces one training batch:

    presence mask -> concentrations -> noisy ligand coordinates -> energies

Affinity kernel (Gaussian): E_open = E_base + E_max * (1 - exp(-d^2 / lambda^2))
where d is the distance in latent space between a ligand and a binding site, and
lambda = kernel_params[0] is the kernel width.

Classic model (use_interface_model=False)
    One binding site per gene product. Learnable: unit_latent (U, D),
    base_energy_u (U,), max_energy_u_raw (U,). sample_batch -> E_open (B, L, U).

Interface model (use_interface_model=True)
    Each unit has a (+) and a (-) face; a binding pocket sits at the interface
    between adjacent units on the ring, so pocket i = 0.5*(v_plus[u_i] +
    v_minus[u_{i+1 mod k}]). sample_batch(batch_size, receptor_indices) ->
    E_open (B, L, R, k_sub), already gathered per interface.

Presence sampling, hierarchical source -> ligand. ZTP = zero-truncated Poisson.
Ligands are partitioned into K = n_presence_blocks source blocks by a partition
seeded from (n_ligands, K). Per sniff:
    1. n_src ~ ZTP(mu_sources, max=K); pick n_src distinct blocks uniformly.
    2. per active block k of size m_k: n_lig ~ ZTP(mu_ligands_per_source,
       max=m_k); pick n_lig distinct ligands within the block uniformly.
    3. every row has at least one ligand by construction.

Numerical notes:
  - ||a-b||^2 = ||a||^2 + ||b||^2 - 2<a,b> avoids the (B, L, U, D) broadcast.
  - UniformNBall: direction * radius^(1/D) is uniform in the N-ball.
  - base_energy init = E[ln c], so EC50 is a typical concentration at step 0.
  - Gumbel-top-k: argsort of Gumbel noise gives a uniform subset draw.
"""
import math

import torch
import torch.distributions as dist
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

class UniformNBall:
    """
    Custom differentiable sampler for an N-dimensional solid sphere (N-ball).
    """
    def __init__(self, loc: torch.Tensor, radius: float, dim: int):
        self.loc = loc          # Shape: (Batch, latent_dim)
        self.radius = radius    # Scalar
        self.dim = dim          # Scalar (latent_dim)

    def rsample(self) -> torch.Tensor:
        # 1. Sample a perfectly random direction
        direction = torch.randn_like(self.loc)
        direction = torch.nn.functional.normalize(direction, p=2, dim=-1)

        # 2. Sample the radius to ensure uniform volume density
        # Draw U ~ Uniform(0, 1)
        u = torch.rand(self.loc.shape[0], 1, device=self.loc.device)
        r = self.radius * (u ** (1.0 / self.dim))

        # 3. Scale and shift
        return self.loc + (direction * r)


class LogNormalConcentration(nn.Module):
    """
    Classic Biophysics assumption: c spans orders of magnitude.
    log10(c) ~ Normal(mu, sigma)
    """
    def __init__(self, n_ligands: int, init_mean: float, init_scale: float):
        super().__init__()

        mu_tensor = torch.tensor(init_mean, dtype=torch.float32)
        if mu_tensor.ndim == 0:
            mu_tensor = torch.ones(n_ligands) * mu_tensor

        sigma_tensor = torch.tensor(init_scale, dtype=torch.float32)
        if sigma_tensor.ndim == 0:
            sigma_tensor = torch.ones(n_ligands) * sigma_tensor

        self.register_buffer('mu', mu_tensor)
        self.register_buffer('log_sigma', torch.log(sigma_tensor))

    def sample(self, batch_size):
        # Sample Log-Space for the whole batch and all ligands
        batch_mu = self.mu.unsqueeze(0).expand(batch_size, -1)
        batch_sigma = torch.exp(self.log_sigma).unsqueeze(0).expand(batch_size, -1)

        dist_log = dist.Normal(batch_mu, batch_sigma)
        log_c = dist_log.rsample()

        # Convert to Real-Space
        return torch.pow(10.0, log_c)

    def get_expected_log_c(self):
        # mu is log10(c). To get natural log ln(c): ln(c) = log10(c) * ln(10)
        return self.mu * math.log(10.0)






class LigandEnvironment(nn.Module):
    def __init__(self, n_genes: int, n_families: int, conc_model,
                 n_ligands: int, observation_noise_sigma: float,
                 latent_dim: int, family_spread: float, distribution_type: str,
                 avg_family_distance: float, n_presence_blocks: int,
                 mu_sources: float, mu_ligands_per_source: float,
                 kernel_params: list = None,
                 use_interface_model: bool = False, block_shared_conc_mean: bool = False,
                 shared_unit_energies: bool = False,
                 fixed_max_energy: Optional[float] = None,
                 family_weights: Optional[list] = None):
        """
        Args:
            n_genes: Number of gene units.
            n_families: Number of ligand families.
            conc_model: a LogNormalConcentration.
            n_ligands: Size of the fixed ligand pool.
            observation_noise_sigma: Observation noise magnitude.
            latent_dim: Dimensionality of the chemical latent space.
            family_spread: Fixed spatial spread of ligand families.
            distribution_type: 'gaussian' or 'uniform'.
            avg_family_distance: Target average Euclidean distance between families.
            n_presence_blocks: K source blocks (deterministic partition from
                               (n_ligands, K)).  K=1 means one source, K=n_ligands
                               means each ligand is its own source.
            mu_sources: Poisson rate for the zero-truncated draw of active sources
                        per sniff (capped at K).
            mu_ligands_per_source: Poisson rate for the zero-truncated draw of
                                   active ligands per source (capped at block size).
            kernel_params: [lambda], the width of the Gaussian affinity kernel.
            use_interface_model: When True, each unit has two faces (+/−) and
                                 binding energy is computed at the interface between
                                 adjacent subunits in the ring.
            block_shared_conc_mean: When True and n_presence_blocks > 1, all ligands
                                    in the same block share one concentration mean.
            family_weights: probability that a pool ligand belongs to each family
                            (length n_families). None draws families uniformly.
        """
        super().__init__()
        self.n_genes = n_genes
        self.n_families = n_families
        self.n_ligands = n_ligands
        self.latent_dim = latent_dim
        self.family_spread = family_spread
        self.avg_family_distance = avg_family_distance
        self.mu_sources = mu_sources
        self.mu_ligands_per_source = mu_ligands_per_source
        self.observation_noise_sigma = observation_noise_sigma
        self.kernel_params = kernel_params if kernel_params is not None else []
        self.use_interface_model = use_interface_model
        # One learnable (E_base, E_max) pair for the WHOLE array instead of one per
        # gene. The analytics in analytics/homomer_array_mi.tex assumes this (A6):
        # with per-gene energies the closed form is only the special case in which
        # the optimiser happens to make them equal. Not implemented for the
        # interface model, whose +/- faces would need a second convention.
        self.shared_unit_energies = shared_unit_energies
        if shared_unit_energies and use_interface_model:
            raise NotImplementedError(
                "shared_unit_energies is implemented for the classic per-unit model "
                "only; use_interface_model splits each unit into +/- faces.")
        # Fixed E_max: one value for every gene, stored as a buffer so it is saved
        # with the model but never optimised.
        self.fixed_max_energy = fixed_max_energy
        self.family_weights = family_weights
        if family_weights is not None and len(family_weights) != n_families:
            raise ValueError(f"family_weights has {len(family_weights)} entries, "
                             f"expected n_families = {n_families}")
        if fixed_max_energy is not None and use_interface_model:
            raise NotImplementedError(
                "fixed_max_energy is implemented for the classic model only.")
        # Composition binding (see bind_receptors). None → the original gather path.
        self.composition = None        # (n_sources, R)
        self.pocket_pairs = None       # (P, 2) interface only: distinct (+face, -face) genes
        self._bound_receptors = None
        self.n_presence_blocks = n_presence_blocks
        self.block_shared_conc_mean = block_shared_conc_mean

        # ----------------------------------------------------------------------
        # PRESENCE BLOCK PARTITION (independent of ligand families)
        # Seeded deterministically from (n_ligands, n_presence_blocks).
        # ----------------------------------------------------------------------
        _g = torch.Generator()
        _g.manual_seed(n_ligands * 131071 + n_presence_blocks)  # 131071 is Mersenne prime
        _perm = torch.randperm(n_ligands, generator=_g)

        _block_size = n_ligands // n_presence_blocks
        _rem = n_ligands % n_presence_blocks
        _sizes = torch.full((n_presence_blocks,), _block_size, dtype=torch.long)
        _sizes[:_rem] += 1  # first _rem blocks get one extra ligand
        _block_ids_sorted = torch.repeat_interleave(
            torch.arange(n_presence_blocks, dtype=torch.long), _sizes
        )
        _block_ids = torch.empty(n_ligands, dtype=torch.long)
        _block_ids[_perm] = _block_ids_sorted
        self.register_buffer('presence_block_id', _block_ids)

        # Hierarchical sampler helper buffers.
        # _block_members: (K, max_m) padded member indices; _block_mask: (K, max_m) validity.
        _max_m = int(_sizes.max().item())
        _block_members = torch.zeros(n_presence_blocks, _max_m, dtype=torch.long)
        _block_valid   = torch.zeros(n_presence_blocks, _max_m, dtype=torch.bool)
        for _k in range(n_presence_blocks):
            _idx = (_block_ids == _k).nonzero(as_tuple=False).squeeze(1)
            _block_members[_k, :len(_idx)] = _idx
            _block_valid[_k,   :len(_idx)] = True
        self.register_buffer('_block_members', _block_members)  # (K, max_m)
        self.register_buffer('_block_valid',   _block_valid)    # (K, max_m)
        self.register_buffer('_block_sizes',   _sizes)          # (K,)

        # Precomputed buffers for vectorized _sample_masks (no Python loop, no GPU-CPU syncs).
        _safe = _block_members.clone()
        _safe[~_block_valid] = n_ligands                         # invalid slots → dummy column L
        self.register_buffer('_safe_block_members', _safe)       # (K, max_m)

        _s_vals = torch.arange(1, n_presence_blocks + 1, dtype=torch.float32)
        _log_pmf_s = _s_vals * math.log(mu_sources) - torch.lgamma(_s_vals + 1)
        _log_pmf_s -= _log_pmf_s.logsumexp(0)
        self.register_buffer('_log_pmf_source', _log_pmf_s)      # (K,)

        _l_vals = torch.arange(1, _max_m + 1, dtype=torch.float32)
        _log_pmf_l = (_l_vals * math.log(mu_ligands_per_source)
                      - torch.lgamma(_l_vals + 1)).unsqueeze(0).expand(n_presence_blocks, -1).clone()
        _log_pmf_l[~_block_valid] = float('-inf')
        _log_pmf_l -= _log_pmf_l.logsumexp(dim=1, keepdim=True)
        self.register_buffer('_log_pmf_ligand', _log_pmf_l)      # (K, max_m)

        # Sparse allocation width. A high-water mark, not a support bound and not a
        # distributional guess: it starts at the minimum and _sample_masks grows it to
        # the largest mixture ever realized, converging to the tightest width that
        # never truncates, for any presence model (including the nonuniform-weight
        # HOOKs in _sample_masks, which no closed-form heuristic would survive).
        # It replaces tail(mu_sources) * _max_m, which took a QUANTILE of the source
        # count but the WORST CASE of the ligands-per-source count, and so collapsed
        # to the whole pool whenever K was not much larger than that quantile. One
        # block of 50k ligands allocated 50k slots per sniff to hold a single ligand.
        self.s_upper = 1

        if distribution_type not in ['gaussian', 'uniform', 'shell']:
            raise ValueError("distribution_type must be 'gaussian', 'uniform', or 'shell'")
        self.distribution_type = distribution_type

        # 1. Inject the Concentration Strategy
        self.concentration_model = conc_model

        # Optional block-shared concentration mean.
        # Turbulent transport: co-occurrence is preserved but concentration ratios are not —
        # only the *mean* is shared per block; per-sniff concentrations draw independently.
        # NOTE: conc_model.mu may already be on CUDA; _block_ids built on CPU, so resolve
        # the target device from conc_model.mu and move before arithmetic.
        if block_shared_conc_mean and n_presence_blocks > 1:
            with torch.no_grad():
                _mu  = conc_model.mu.clone()
                _dev = _mu.device
                _ids = _block_ids.long().to(_dev)
                # bincount is CPU-only; compute there, then move.
                _block_cnt = torch.bincount(
                    _block_ids.long(), minlength=n_presence_blocks
                ).float().to(_dev)
                _block_sum = torch.zeros(
                    n_presence_blocks, dtype=_mu.dtype, device=_dev
                ).scatter_add(0, _ids, _mu)
                _block_mean = _block_sum / _block_cnt
                conc_model.mu.copy_(_block_mean[_ids])

        # ----------------------------------------------------------------------
        # MECHANISTIC LATENT SPACE INITIALIZATION
        # ----------------------------------------------------------------------

        _init_val = math.log(math.e ** 10.0 - 1.0)   # softplus(x) ≈ 10 at init

        if use_interface_model:
            # Each unit has two faces; pockets form at adjacent (+)/(−) interfaces.
            self.unit_latent_plus  = nn.Parameter(torch.randn(n_genes, latent_dim) * 1.0)
            self.unit_latent_minus = nn.Parameter(torch.randn(n_genes, latent_dim) * 1.0)
            self.max_energy_u_raw_plus  = nn.Parameter(torch.full((n_genes,), _init_val))
            self.max_energy_u_raw_minus = nn.Parameter(torch.full((n_genes,), _init_val))
        else:
            # Classic per-unit model.
            self.unit_latent = nn.Parameter(torch.randn(n_genes, latent_dim) * 1.0)
            if fixed_max_energy is not None:
                raw = math.log(math.expm1(fixed_max_energy))   # inverse softplus
                self.register_buffer('max_energy_u_raw_fixed', torch.full((1,), raw))
            elif shared_unit_energies:
                self.max_energy_u_raw_shared = nn.Parameter(torch.full((1,), _init_val))
            else:
                self.max_energy_u_raw = nn.Parameter(torch.full((n_genes,), _init_val))

        # 2. The Environment is Fixed (Family Prototype Coordinates)
        fixed_families = self._generate_family_centers(n_families, latent_dim)
        self.register_buffer('family_latent', fixed_families)

        # 2.5 Ligand Pool Initialization
        with torch.no_grad():
            # Assign each ligand to a family randomly
            if family_weights is None:
                ligand_family_assignments = torch.randint(0, n_families, (n_ligands,))
            else:
                w = torch.as_tensor(family_weights, dtype=torch.float)
                ligand_family_assignments = torch.multinomial(w / w.sum(), n_ligands,
                                                              replacement=True)
            self.register_buffer('ligand_family_assignments', ligand_family_assignments)

            # Draw permanent base coordinates for the ligands
            base_centers = fixed_families[ligand_family_assignments]
            self.register_buffer('ligand_latent', self.sample_near_centers(base_centers))

        # 3. Unit-specific Base Energies
        # E_base = E_o(u, ℓ_opt): open-state energy at the optimally matched ligand.
        # Initialised to E[ln c] so EC50 matches the expected concentration at start.
        global_avg_log_c = self.concentration_model.get_expected_log_c().mean().item()
        if use_interface_model:
            self.base_energy_u_plus  = nn.Parameter(torch.ones(n_genes) * global_avg_log_c)
            self.base_energy_u_minus = nn.Parameter(torch.ones(n_genes) * global_avg_log_c)
        elif shared_unit_energies:
            self.base_energy_u_shared = nn.Parameter(torch.full((1,), global_avg_log_c))
        else:
            self.base_energy_u = nn.Parameter(torch.ones(n_genes) * global_avg_log_c)

    # ----------------------------------------------------------------------
    # Energy accessors (classic model).
    #
    # Always return a (n_genes,) tensor so every caller can index it by gene,
    # whether the underlying parameter is per-gene or a single shared scalar.
    # expand() returns a view, so autograd accumulates the SUM over genes into
    # the scalar, which is exactly the gradient of a genuinely shared parameter
    # rather than an average of per-gene gradients.
    # ----------------------------------------------------------------------
    def base_energies(self) -> torch.Tensor:
        """Per-gene baseline energy E_base, shape (n_genes,)."""
        if self.shared_unit_energies:
            return self.base_energy_u_shared.expand(self.n_genes)
        return self.base_energy_u

    def max_energies_raw(self) -> torch.Tensor:
        """Pre-softplus E_max, shape (n_genes,). Gaussian kernel only."""
        if getattr(self, 'fixed_max_energy', None) is not None:
            return self.max_energy_u_raw_fixed.expand(self.n_genes)
        if self.shared_unit_energies:
            return self.max_energy_u_raw_shared.expand(self.n_genes)
        return self.max_energy_u_raw



    def sample_near_centers(self, centers: torch.Tensor) -> torch.Tensor:
        """Draw one latent point per row of `centers` from the family distribution.

        The single definition of what "a member of this family" means in latent
        space.  Ligand placement uses it at construction; analysis reuses it to
        sample the family clouds it draws, so a plotted cloud cannot describe a
        different distribution from the one the ligands were drawn from.

        Callers that must not perturb the global RNG should wrap the call in
        torch.random.fork_rng().
        """
        if self.distribution_type == 'gaussian':
            return dist.Normal(loc=centers, scale=self.family_spread).rsample()
        if self.distribution_type == 'uniform':
            return UniformNBall(loc=centers, radius=self.family_spread,
                                dim=self.latent_dim).rsample()
        if self.distribution_type == 'shell':
            # Defeats high-D concentration-of-measure: direction uniform on sphere,
            # radius uniform in [0, family_spread] — equal weight at every shell radius.
            direction = torch.nn.functional.normalize(
                torch.randn_like(centers), p=2, dim=-1)
            r = torch.rand(centers.shape[0], 1, device=centers.device) * self.family_spread
            return centers + direction * r
        raise ValueError(f"Unsupported family distribution: {self.distribution_type}")

    def _generate_family_centers(self, n_families: int, latent_dim: int) -> torch.Tensor:
        """
        Default behavior: Prototype centers spread inside a uniform N-ball,
        with the radius calculated to yield the requested average distance.
        """
        # 1. Empirically find the average distance in a unit N-ball
        with torch.no_grad():
            loc_mc = torch.zeros(20000, latent_dim)
            unit_sampler = UniformNBall(loc=loc_mc, radius=1.0, dim=latent_dim)
            p1, p2 = unit_sampler.rsample(), unit_sampler.rsample()
            unit_avg_dist = torch.norm(p1 - p2, dim=-1).mean().item()

        # 2. Scale radius to match target average distance
        target_radius = self.avg_family_distance / unit_avg_dist

        loc_centers = torch.zeros(n_families, latent_dim)
        sampler = UniformNBall(loc=loc_centers, radius=target_radius, dim=latent_dim)
        return sampler.rsample()

    @property
    def interaction_mu(self) -> torch.Tensor:
        """
        Open-state energy for each unit against each ligand's exact center.
        Shape: (n_genes, n_ligands).

        Classic model only.  For the interface model call interaction_mu_interface().
        """
        diff = self.unit_latent.unsqueeze(1) - self.ligand_latent.unsqueeze(0)  # (U, L, D)
        dist_sq = (diff ** 2).sum(dim=-1)  # (U, L)
        base_e = self.base_energies()                  # (U,)
        max_e = F.softplus(self.max_energies_raw())  # (U,)
        lambda_sq = self.kernel_params[0] ** 2
        return base_e.unsqueeze(1) + max_e.unsqueeze(1) * (1.0 - torch.exp(-dist_sq / lambda_sq))


    def _sample_masks(self, batch_size: int) -> torch.Tensor:
        """(B, L) float presence mask via the vectorized hierarchical source→ligand sampler.

        All K blocks are processed simultaneously without a Python loop. When the
        usual sparse width is below L, one scalar GPU→CPU synchronization sizes
        the output so rare large mixtures are never silently truncated.

          1. Draw n_src ~ ZTP(mu_sources, K) via Gumbel-max; select n_src blocks via Gumbel-top-k.
          2. Draw n_lig[b,k] ~ ZTP(mu_ligands_per_source, m_k) for all (b,k) via Gumbel-max.
          3. Select top-n_lig[b,k] ligands within each block via Gumbel-top-k (argsort + scatter).
          4. scatter_add_ into (B, L+1) buffer; dummy column L absorbs padding from
             _safe_block_members (invalid slots pre-redirected to index L in __init__).

        # HOOK: the two uniform selections (sources and ligands-per-source) are the
        # only place where per-block or per-ligand frequency weights would re-enter.
        # Replace _log_pmf_source with log(w_k) or _log_pmf_ligand with log(w_l) to
        # implement nonuniform sampling over sources or ligands within a source.
        """
        device = self.presence_block_id.device
        B, K, L = batch_size, self.n_presence_blocks, self.n_ligands
        max_m = self._block_members.shape[1]

        # --- Step 1: ZTP draw of n_src, Gumbel-top-K to select active source blocks ---
        # HOOK: replace _log_pmf_source with log(w_block) for nonuniform source weights.
        n_src = torch.multinomial(self._log_pmf_source.exp(), B, replacement=True) + 1  # (B,)

        gumbel_s = -torch.log(-torch.log(torch.rand(B, K, device=device).clamp(1e-9, 1 - 1e-9)))
        rank_s = gumbel_s.argsort(dim=1, descending=True)
        active_src = torch.arange(K, device=device).unsqueeze(0) < n_src.unsqueeze(1)
        active_src_mask = torch.zeros(B, K, dtype=torch.bool, device=device)
        active_src_mask.scatter_(1, rank_s, active_src)                          # (B, K)

        # --- Step 2: ZTP draw of n_lig for every (b, k) simultaneously ---
        # Gumbel-max trick: argmax(log_pmf + Gumbel) ~ categorical(exp(log_pmf)).
        # HOOK: replace _log_pmf_ligand with log(w_ligand_k) for nonuniform ligand weights.
        gumbel_ztp = (self._log_pmf_ligand.unsqueeze(0)
                      + (-torch.log(-torch.log(
                          torch.rand(B, K, max_m, device=device).clamp(1e-9, 1 - 1e-9)
                      ))))                                                        # (B, K, max_m)
        n_lig = gumbel_ztp.argmax(dim=-1) + 1                                   # (B, K) in [1, m_k]
        n_lig = n_lig * active_src_mask                                          # 0 for inactive blocks

        # --- Step 3: Gumbel-top-n_lig within each block, all (b, k) in parallel ---
        gumbel_l = -torch.log(-torch.log(
            torch.rand(B, K, max_m, device=device).clamp(1e-9, 1 - 1e-9)
        ))
        gumbel_l.masked_fill_(~self._block_valid.unsqueeze(0), float('-inf'))    # invalid slots
        gumbel_l.masked_fill_(~active_src_mask.unsqueeze(-1), float('-inf'))     # inactive blocks

        rank_l = gumbel_l.argsort(dim=-1, descending=True)                      # (B, K, max_m)
        picked_ranked = torch.arange(max_m, device=device) < n_lig.unsqueeze(-1)# (B, K, max_m)
        picked_mask = torch.zeros(B, K, max_m, dtype=torch.bool, device=device)
        picked_mask.scatter_(2, rank_l, picked_ranked)                           # top-n_lig → True

        # --- Step 4: scatter into M; invalid slots point to dummy column L ---
        picked_flat = picked_mask.view(B, K * max_m).float()
        idx_flat    = self._safe_block_members.view(1, K * max_m).expand(B, -1) # (B, K*max_m)
        M = torch.zeros(B, L + 1, device=device)
        M.scatter_add_(1, idx_flat, picked_flat)

        # --- Step 5: retain ALL selected ligands, widening rare overflow batches ---
        # torch.where: real member index where picked, dummy L otherwise.
        safe_flat    = self._safe_block_members.view(K * max_m)                 # (K*max_m,)
        selected_flat = torch.where(picked_flat.bool(), safe_flat.unsqueeze(0), L)  # (B, K*max_m)
        # topk on picked_flat (1.0 = selected) brings selected positions to front without sorting.
        # s_upper is a high-water mark: grow it on overflow, never shrink. Monotone
        # growth keeps the output shape stable at one allocator block size instead of
        # oscillating with the per-batch maximum.
        width = self.s_upper
        if width < L:
            mx = int(n_lig.sum(dim=1).max().item())
            if mx > width:
                width = self.s_upper = mx
        _, front_order = picked_flat.topk(width, dim=-1, sorted=False)     # (B, width)
        sparse_idx = selected_flat.gather(1, front_order)                  # (B, width)

        return M[:, :L].clamp(max=1.0), sparse_idx

    def _sample_noisy_ligands(self, batch_size: int, sparse_idx: torch.Tensor) -> torch.Tensor:
        """(B, sparse_width, D) — noisy coordinates for the selected ligands only.

        sparse_idx: (B, s_upper) long tensor — real ligand indices [0, L-1] for present
                    ligands, n_ligands (dummy) for padding slots.
        Dummy slots are zeroed out after noise injection so they contribute nothing to
        downstream energies (their concentrations are also 0 in sample_batch).
        """
        S = sparse_idx.shape[1]
        safe_idx = sparse_idx.clamp(max=self.n_ligands - 1)                  # (B, S)
        base     = self.ligand_latent[safe_idx]                               # (B, S, D)
        noise    = (torch.randn(batch_size, S, self.latent_dim, device=base.device)
                    * self.observation_noise_sigma)
        dummy    = (sparse_idx == self.n_ligands).unsqueeze(-1)               # (B, S, 1)
        return (base + noise).masked_fill(dummy, 0.0)                         # (B, S, D)

    # ------------------------------------------------------------------
    # Composition binding — the "compute each distinct thing once" fast path
    # ------------------------------------------------------------------

    def bind_receptors(self, receptor_indices: torch.Tensor):
        """Precompute the (n_sources, R) composition matrix for a receptor array.

        A "source" is a distinct thing whose interaction energy has to be computed:

          classic model    → a GENE.  n_sources = n_genes.  composition[u, r] is the
                             number of receptor r's k_sub slots filled by gene u.
          interface model  → a POCKET, identified by the (plus-face gene, minus-face
                             gene) pair straddling it.  A pocket's energy depends ONLY
                             on that pair — not on which receptor it sits in, nor on
                             its position in the ring — so there are at most n_genes^2
                             distinct pockets no matter how large the array is.
                             composition[p, r] counts how often pocket p occurs in r.

        Binding lets sample_batch() evaluate each distinct source once and lets
        physics turn (B, L, n_sources) into (B, L, R) with a single matmul.  In the
        interface model the redundancy removed is large: a pool of 53k receptors uses
        267k pocket slots drawn from only ~644 distinct pockets (~400x).

        Sets self.composition and (interface only) self.pocket_pairs, and returns the
        composition matrix.  Call unbind() to restore the original gather path.
        """
        ref = (self.unit_latent_plus if self.use_interface_model else self.unit_latent)
        ri = receptor_indices.to(ref.device)
        R, k = ri.shape

        if not self.use_interface_model:
            n_sources = self.n_genes
            src = ri.reshape(-1)                                        # (R*k,)
            self.pocket_pairs = None
        else:
            idx_i = ri                                                  # (R, k)
            idx_j = ri.roll(-1, dims=1)                                 # (R, k) cyclic
            pair_id = (idx_i * self.n_genes + idx_j).reshape(-1)        # (R*k,)
            uniq, inverse = torch.unique(pair_id, return_inverse=True)
            n_sources = uniq.numel()
            src = inverse                                               # (R*k,)
            self.pocket_pairs = torch.stack(
                [uniq // self.n_genes, uniq % self.n_genes], dim=1)     # (P, 2)

        cols = torch.arange(R, device=ri.device).unsqueeze(1).expand(R, k).reshape(-1)
        comp = torch.zeros(n_sources, R, device=ri.device, dtype=torch.float32)
        comp.index_put_((src, cols),
                        torch.ones(src.numel(), device=ri.device), accumulate=True)
        self.composition = comp
        self._bound_receptors = ri
        return comp


    def _compute_pocket_energies(self, v_ligands: torch.Tensor) -> torch.Tensor:
        """(B, L, P) — one energy per DISTINCT pocket, for the bound receptor array.

        Same formula as the interface branch of _compute_energies, evaluated on the
        deduplicated (plus-face, minus-face) gene pairs instead of on every (receptor,
        ring position) slot.  Bit-for-bit the same values, ~n_genes^2/(R*k_sub) of the work.
        """
        idx_i = self.pocket_pairs[:, 0]                                  # (P,)
        idx_j = self.pocket_pairs[:, 1]                                  # (P,)

        v_pocket = 0.5 * (self.unit_latent_plus[idx_i]
                          + self.unit_latent_minus[idx_j])               # (P, D)
        E_base   = 0.5 * (self.base_energy_u_plus[idx_i]
                          + self.base_energy_u_minus[idx_j])             # (P,)

        a_sq    = (v_ligands ** 2).sum(dim=-1, keepdim=True)             # (B, L, 1)
        b_sq    = (v_pocket ** 2).sum(dim=-1)                            # (P,)
        ab      = torch.einsum('bld,pd->blp', v_ligands, v_pocket)       # (B, L, P)
        dist_sq = (a_sq + b_sq[None, None, :] - 2.0 * ab).clamp(min=0.0) # (B, L, P)

        E_max = 0.5 * (F.softplus(self.max_energy_u_raw_plus[idx_i])
                       + F.softplus(self.max_energy_u_raw_minus[idx_j]))   # (P,)
        lambda_sq = self.kernel_params[0] ** 2
        return (E_base[None, None, :]
                + E_max[None, None, :] * (1.0 - torch.exp(-dist_sq / lambda_sq)))

    def _compute_energies(
        self,
        v_ligands: torch.Tensor,
        receptor_indices: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Open-state energies from noisy ligand coordinates.

        Classic model   → (B, L, U)
        Interface model → (B, L, R, k_sub)

        ‖a−b‖² = ‖a‖²+‖b‖²−2⟨a,b⟩ avoids the (B, L, U, D) broadcast.
        For the interface model, pockets are flattened to (R*k, D) then reshaped.
        """
        if not self.use_interface_model:
            # ------------------------------------------------------------------
            # Classic path: (B, L, U)
            # ------------------------------------------------------------------
            a_sq    = (v_ligands ** 2).sum(dim=-1, keepdim=True)               # (B, L, 1)
            b_sq    = (self.unit_latent ** 2).sum(dim=-1)                      # (U,)
            ab      = torch.einsum('bld,ud->blu', v_ligands, self.unit_latent) # (B, L, U)
            dist_sq = (a_sq + b_sq[None, None, :] - 2.0 * ab).clamp(min=0.0)  # (B, L, U)
            base_e = self.base_energies()                                     # (U,)
            max_e     = F.softplus(self.max_energies_raw())                # (U,)
            lambda_sq = self.kernel_params[0] ** 2
            return (base_e[None, None, :]
                    + max_e[None, None, :] * (1.0 - torch.exp(-dist_sq / lambda_sq)))

        # ----------------------------------------------------------------------
        # Interface path
        # ----------------------------------------------------------------------
        # Bound: evaluate the ~n_genes^2 DISTINCT pockets once → (B, L, P).
        # physics then contracts with the (P, R) composition. Unbound: the original
        # per-slot path → (B, L, R, k_sub).
        if self.composition is not None and self.pocket_pairs is not None:
            return self._compute_pocket_energies(v_ligands)

        if receptor_indices is None:
            raise ValueError(
                "receptor_indices must be supplied to _compute_energies() "
                "when use_interface_model=True."
            )
        B, L, D = v_ligands.shape
        R, k    = receptor_indices.shape

        idx_i = receptor_indices                      # (R, k)
        idx_j = receptor_indices.roll(-1, dims=1)     # (R, k) — next unit (cyclic)

        # Pocket embedding: average of adjacent +/− faces
        # (observation noise is on the ligand side only, matching the classic model)
        v_pocket      = 0.5 * (self.unit_latent_plus[idx_i]
                                + self.unit_latent_minus[idx_j])              # (R, k, D)
        E_base_pocket = 0.5 * (self.base_energy_u_plus[idx_i]
                                + self.base_energy_u_minus[idx_j])            # (R, k)

        # Flatten pockets to (R*k, D), compute (B, L, R*k), then reshape
        b       = v_pocket.reshape(R * k, D)                                   # (R*k, D)
        a_sq    = (v_ligands ** 2).sum(dim=-1, keepdim=True)                   # (B, L, 1)
        b_sq    = (b ** 2).sum(dim=-1)                                         # (R*k,)
        ab      = torch.einsum('bld,kd->blk', v_ligands, b)                   # (B, L, R*k)
        dist_sq = (a_sq + b_sq[None, None, :] - 2.0 * ab).clamp(min=0.0)     # (B, L, R*k)
        dist_sq = dist_sq.view(B, L, R, k)                                    # (B, L, R, k)

        E_max_pocket = 0.5 * (
            F.softplus(self.max_energy_u_raw_plus[idx_i])
            + F.softplus(self.max_energy_u_raw_minus[idx_j])
        )  # (R, k)
        lambda_sq = self.kernel_params[0] ** 2
        return (E_base_pocket[None, None, :, :]
                + E_max_pocket[None, None, :, :] * (1.0 - torch.exp(-dist_sq / lambda_sq)))

    def sample_batch(
        self,
        batch_size: int,
        receptor_indices: Optional[torch.Tensor] = None,
        return_dense_conc: bool = False,
    ):
        """Generate one batch of sensory environments.

        Args:
            batch_size:       Number of samples B.
            receptor_indices: (R, k_sub) ring layout; required when
                              use_interface_model=True.

        Returns (classic model):
            E_open:        (B, s_upper, U) — open-state energy for present ligands only.
            concs:         (B, s_upper)    — concentrations for present ligands (0 for padding).
            mixture_masks: (B, L)          — full presence mask (needed for MI metrics).

        Returns (interface model):
            E_open:        (B, s_upper, R, k_sub)
            concs:         (B, s_upper)
            mixture_masks: (B, L)

        The sparse width s_upper starts at 1 and is a high-water mark: the returned
        ligand axis widens to the actual maximum count whenever a batch exceeds it, and
        that width is retained. No selected ligands are ever discarded. Padding slots carry index L
        with concentration 0, contributing exp(-27) ≈ 0 to the logsumexp in physics.
        The hierarchical sampler guarantees S >= 1 for every row by construction.
        """
        masks, sparse_idx = self._sample_masks(batch_size)                    # (B,L), (B,s_upper)
        assert masks.sum(-1).min() >= 1, "hierarchical sampler produced an empty row"

        # Sparse concentrations: gather only the present ligands.
        safe_sidx = sparse_idx.clamp(max=self.n_ligands - 1)                  # (B, s_upper)
        concs_all = self.concentration_model.sample(batch_size)                # (B, L)
        concs     = concs_all.gather(1, safe_sidx).masked_fill(
            sparse_idx == self.n_ligands, 0.0
        )                                                                       # (B, s_upper)

        v_ligands = self._sample_noisy_ligands(batch_size, sparse_idx)        # (B, s_upper, D)
        E_open    = self._compute_energies(v_ligands, receptor_indices)        # (B, s_upper, *)

        if return_dense_conc:
            # (B, L) per-ligand concentration aligned to ligand identity (0 if absent),
            # built from the same sparse_idx/concs that produced `activity`.  Padding
            # slots carry index L and scatter into a throwaway dummy column.
            concs_dense = torch.zeros(batch_size, self.n_ligands + 1,
                                      device=concs.device, dtype=concs.dtype)
            concs_dense.scatter_(1, sparse_idx, concs)
            return E_open, concs, masks, concs_dense[:, :self.n_ligands]

        return E_open, concs, masks


