"""Family moves: jump a receptor to a random family, accept by Metropolis on MI.

Homomers only, so receptor r is gene r and its position is env.unit_latent[r].
A move redraws the position as a member of a uniformly chosen family, drawn the
way that family's ligands are. E_base is shared and left as is.

Acceptance: dMI = MI_after - MI_before on one fixed batch, accepted with
probability min(1, exp(dMI / T_move)). T_move starts at the std of dMI over R
trial moves and falls linearly to 0 over the annealing window, like the
receptor steepness.

relax=True refines the moved receptor alone, others frozen, for as many steps
as separate two moves, before comparing. A rejection then costs compute only.
"""
import math

import torch


def place_near_families(env, genes):
    """Redraw the positions of `genes` near uniformly chosen family centres."""
    device = env.family_latent.device
    genes = torch.as_tensor(genes, device=device)
    families = torch.randint(0, env.family_latent.shape[0], (len(genes),), device=device)
    with torch.no_grad():
        env.unit_latent[genes] = env.sample_near_centers(env.family_latent[families])


class FamilyMoves:
    def __init__(self, runner, env, physics, loss_fn, optimizer, anneal_epochs):
        cfg = runner.config
        self.runner, self.env, self.physics = runner, env, physics
        self.loss_fn, self.optimizer = loss_fn, optimizer
        self.relax = cfg.family_move_relax
        self.anneal_epochs = anneal_epochs
        self.every = max(1, anneal_epochs // (cfg.family_moves * env.n_genes))
        self.accepted = 0
        trials = [self._delta(self._jump(), relax=False, restore=True)
                  for _ in range(env.n_genes)]
        self.T0 = max(torch.tensor(trials).std().item(), 1e-3)

    def due(self, epoch):
        return 0 < epoch < self.anneal_epochs and epoch % self.every == 0

    def step(self, epoch):
        """One proposal at the current steepness."""
        gene, old, seed, before = self._jump()
        dmi = self._delta((gene, old, seed, before), relax=self.relax, restore=False)
        T = self.T0 * (1.0 - epoch / self.anneal_epochs)
        if dmi >= 0 or torch.rand(()).item() < math.exp(dmi / T):
            self.accepted += 1
            # Momentum from the old site is meaningless here. The squared-gradient
            # average is kept: its scale does not depend on where the receptor sits.
            state = self.optimizer.state.get(self.env.unit_latent)
            if state:
                state["exp_avg"][gene] = 0.0
        else:
            with torch.no_grad():
                self.env.unit_latent[gene] = old

    # -- internals ---------------------------------------------------------

    def _jump(self):
        """Move one random receptor. Returns what is needed to judge or undo it."""
        gene = torch.randint(0, self.env.n_genes, ()).item()
        seed = torch.randint(0, 2 ** 62, ()).item()
        before = self._mi(seed)
        old = self.env.unit_latent[gene].detach().clone()
        place_near_families(self.env, [gene])
        return gene, old, seed, before

    def _delta(self, jump, relax, restore):
        gene, old, seed, before = jump
        if relax:
            self._refine(gene)
        dmi = self._mi(seed) - before
        if restore:
            with torch.no_grad():
                self.env.unit_latent[gene] = old
        return dmi

    def _mi(self, seed):
        """MI estimate on the batch fixed by `seed`, so two calls see the same sniffs."""
        device = self.env.unit_latent.device
        r = self.runner
        with torch.no_grad(), torch.random.fork_rng(
                devices=[device.index or 0] if device.type == "cuda" else []):
            torch.manual_seed(seed)
            energies, concs, _ = self.env.sample_batch(r.config.batch_size)
            activity = r._activity(self.physics, energies, concs, r.receptor_indices,
                                   self.env).float()
            return -self.loss_fn(activity).item()

    def _refine(self, gene):
        """Gradient steps on one receptor's position. Everything else is frozen.

        A fresh Adam sees zero gradient on the other rows, so it leaves them
        exactly in place, and the main optimizer's state is never touched.
        """
        r = self.runner
        opt = torch.optim.Adam([self.env.unit_latent], lr=r.config.lr)
        for _ in range(self.every):
            self.env.zero_grad(set_to_none=True)
            energies, concs, _ = self.env.sample_batch(r.config.batch_size)
            activity = r._activity(self.physics, energies, concs, r.receptor_indices,
                                   self.env).float()
            self.loss_fn(activity).backward()
            grad = self.env.unit_latent.grad
            keep = grad[gene].clone()
            grad.zero_()
            grad[gene] = keep
            opt.step()
        self.env.zero_grad(set_to_none=True)
