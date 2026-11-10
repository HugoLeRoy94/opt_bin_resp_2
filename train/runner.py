"""One optimization: build the world, train the array, measure it once.

The loop is short. Everything around it is schedule and measurement.

Sharpness schedule. The receptor activation starts soft (temperature
initial_temperature) and anneals to `temperature` over the first 80% of epochs,
then holds. Starting sharp gives vanishing gradients, and ending soft would mean
reporting a number for a model that is not the one evaluation uses, so the final
measurement is always taken at the sharp endpoint.
"""
import math

import torch
import torch.optim as optim

from estimate.entropy import (KT_EPS, compute_kt_entropy, compute_kt_upper_entropy,
                              compute_response_conditional_entropy)
from estimate.grouping import CellGrouping, GroupedResponseCounter
from estimate.losses import (ExactMutualInformationLoss,
                             GroupedCellMutualInformationLoss,
                             GroupedKTMutualInformationLoss, KTMutualInformationLoss)
from estimate.measurements import REGISTRY, codeword_entropy, response_counting_metrics
from model.cells import CellArray, CellReadout, cell_activity
from model.environment import LigandEnvironment, LogNormalConcentration
from model.receptors import BinaryReceptor, compute_initial_temperature

KT_TRAIN_TILE = 4096
EVAL_TILE = 2048

# Measurements that need the WHOLE evaluation batch, so the runner streams them
# itself rather than calling the registry entry on one chunk.
_STREAMED = {"entropy_kt", "entropy_kt_upper", "conditional_entropy_response",
             "mutual_information_kt", "mutual_information_kt_upper",
             "codeword_entropy", "mutual_information_counting",
             "grouped_information", "grouped_counting"}


def build_loss(config, grouping=None):
    """The objective picked by (config.mi, config.grouped).

    Two independent axes, so four combinations and no special names:

                     grouped=False                grouped=True
        shannon      ExactMutualInformationLoss   GroupedCellMutualInformationLoss
        kt           KTMutualInformationLoss      GroupedKTMutualInformationLoss
    """
    tile = min(int(config.batch_size), KT_TRAIN_TILE)
    common = dict(collision_chunk_size=tile,
                  recompute_backward=config.recompute_backward,
                  compile_kt=config.compile_kt)
    if config.mi == "shannon":
        return (GroupedCellMutualInformationLoss(grouping, config.cell_grouped_max_states)
                if config.grouped else ExactMutualInformationLoss(**common))
    return (GroupedKTMutualInformationLoss(grouping, **common)
            if config.grouped else KTMutualInformationLoss(**common))


class Runner:
    """Trains one Config and writes its results through `logger`."""

    def __init__(self, config, logger, device=None):
        self.config = config
        self.logger = logger
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.readout = None
        self.grouping = None
        self.grouped_estimator = None
        self._response_generator = None
        # Set by build(). A test script loads a checkpoint into these and calls
        # evaluate(), so measuring never re-implements construction.
        self.env = self.physics = self.loss_fn = self.receptor_indices = None

    # -- construction ------------------------------------------------------

    def build(self, verbose=True):
        """Construct the world, the array and the objective. Sets the attributes."""
        cfg = self.config
        receptor_indices = torch.tensor(cfg.receptor_indices, dtype=torch.long,
                                        device=self.device)

        if cfg.is_cell_mode():
            # The pool order is deterministic, so CellArray rebuilds exactly the
            # pool that Config.__post_init__ stored in receptor_indices.
            cells = CellArray(
                (None if cfg.cell_receptors is not None else cfg.cell_gene_sets),
                cfg.k_sub, stoichiometry=cfg.cell_stoichiometry,
                use_interface_model=cfg.use_interface_model,
                repertoires=cfg.cell_receptors).to(self.device)
            self.readout = CellReadout(cells.W, k_sub=cfg.k_sub).to(self.device)
            self.grouping = CellGrouping(cells.W).to(self.device)
            if verbose:
                print(f"[cells] {self.grouping.n_cells} cells -> "
                      f"{self.grouping.n_groups} groups, "
                      f"{self.grouping.n_states} count states")

        env = LigandEnvironment(
            cfg.n_genes, cfg.n_families,
            conc_model=LogNormalConcentration(cfg.n_ligands, cfg.conc_mean, cfg.conc_std),
            n_ligands=cfg.n_ligands, mu_sources=cfg.mu_sources,
            mu_ligands_per_source=cfg.mu_ligands_per_source,
            observation_noise_sigma=cfg.observation_noise_sigma,
            latent_dim=cfg.latent_dim, family_spread=cfg.family_spread,
            avg_family_distance=cfg.average_family_distance,
            n_presence_blocks=cfg.n_presence_blocks, kernel_params=cfg.kernel_params,
            distribution_type=cfg.distribution_type,
            use_interface_model=cfg.use_interface_model,
            block_shared_conc_mean=cfg.block_shared_conc_mean,
            shared_unit_energies=cfg.shared_unit_energies,
            fixed_max_energy=cfg.fixed_max_energy,
            family_weights=cfg.family_weights).to(self.device)

        physics = BinaryReceptor(cfg.n_genes, cfg.k_sub,
                                 temperature=cfg.temperature).to(self.device)

        # Evaluate each DISTINCT energy source once and contract to receptors with
        # one matmul. Removes the (B, L, R, k_sub) intermediate entirely.
        env.bind_receptors(receptor_indices)
        n_src = env.composition.shape[0]
        if verbose and cfg.use_interface_model:
            print(f"[composition] R={receptor_indices.shape[0]}  "
                  f"{receptor_indices.numel()} pocket slots -> {n_src} distinct "
                  f"({receptor_indices.numel() / max(n_src, 1):.0f}x fewer)")
        elif verbose:
            print(f"[composition] R={receptor_indices.shape[0]}  genes={n_src} "
                  f"(drops the {cfg.k_sub}x-wide gather)")

        loss_fn = build_loss(cfg, self.grouping).to(self.device)
        if isinstance(loss_fn, GroupedCellMutualInformationLoss):
            self.grouped_estimator = loss_fn

        optimizer = optim.Adam(list(env.parameters()) + list(physics.parameters()),
                               lr=cfg.lr)
        self.env, self.physics = env, physics
        self.loss_fn, self.receptor_indices = loss_fn, receptor_indices
        return env, physics, loss_fn, optimizer, receptor_indices

    def _grouped_estimator_for(self, _):
        """The exact count estimator, built once, shared with the training loss."""
        if self.grouped_estimator is None:
            self.grouped_estimator = GroupedCellMutualInformationLoss(
                self.grouping, self.config.cell_grouped_max_states).to(self.device)
        return self.grouped_estimator

    def _activity(self, physics, energies, concs, receptor_indices, env):
        """(B, R) receptor activity, or (B, C) cell activity in cell mode.

        The only place the two pictures differ. Everything downstream takes a
        (B, N) tensor and does not care what N counts.
        """
        comp = env.composition
        if self.readout is None:
            return physics(energies, concs, receptor_indices, composition=comp)
        return cell_activity(physics, self.readout, energies, concs, receptor_indices,
                             pre_gathered=False, chunk_size=self.config.cell_pool_chunk,
                             recompute=self.config.recompute_backward, composition=comp)

    # -- evaluation --------------------------------------------------------

    def _eval_stats(self, env, physics, loss_fn, receptor_indices, batch_size,
                    measurement_fns=None):
        """Measure over batch_size samples, with per-pass memory bounded by chunks.

        Measurement is always float32 and never under autocast: a few thousandths
        of a bit is acceptable training noise but not in a published number.
        """
        cfg = self.config
        chunk = min(cfg.eval_chunk_size or cfg.batch_size, batch_size)
        requested = set(cfg.measurement_fns if measurement_fns is None else measurement_fns)

        native_grouped = isinstance(loss_fn, GroupedCellMutualInformationLoss)
        native_grouped_kt = isinstance(loss_fn, GroupedKTMutualInformationLoss)
        streamed = set(_STREAMED)
        if native_grouped or native_grouped_kt:
            streamed.add("full_array_entropy")

        want_grouped = ("grouped_information" in requested
                        or (native_grouped and "full_array_entropy" in requested))
        want_lower = (bool(requested & {"entropy_kt", "mutual_information_kt"})
                      or (native_grouped_kt and "full_array_entropy" in requested))
        want_upper = bool(requested & {"entropy_kt_upper", "mutual_information_kt_upper"})
        want_counting = "mutual_information_counting" in requested
        want_grouped_counting = "grouped_counting" in requested
        want_hard = "codeword_entropy" in requested
        grouped_est = self._grouped_estimator_for(loss_fn) if want_grouped else None
        counter = GroupedResponseCounter(self.grouping) if want_grouped_counting else None

        ri_batch = receptor_indices if env.use_interface_model else None
        stat = {}
        with torch.no_grad():
            E, concs, _ = env.sample_batch(chunk, receptor_indices=ri_batch)
            activity = self._activity(physics, E, concs, receptor_indices, env)

            # Chunk-local measurements: everything not in `streamed`.
            for name in requested - streamed:
                fn = REGISTRY[name]
                out = fn(activity, loss_fn) if name in (
                    "full_array_entropy", "entropy_kt", "entropy_kt_upper",
                    "grouped_information", "grouped_counting") else fn(activity)
                stat.update(out) if isinstance(out, dict) else stat.update({name: out})

            if not (requested & streamed):
                # Chunk-local measurements run on ONE forward pass, so they use
                # `chunk` samples however large batch_size was. Recording it
                # stops a 2^26 budget being read into a number that saw 512.
                stat["response_evaluation_samples"] = activity.shape[0]
                return stat

            if (want_counting or want_grouped_counting) and self._response_generator is None:
                # Output sampling must not perturb the world's or training RNG.
                self._response_generator = torch.Generator(device=activity.device)
                self._response_generator.manual_seed((torch.initial_seed() + 104729) % (2 ** 63))

            acts, hard_codes, sampled_codes = [], [], []
            cond_sum = torch.zeros((), dtype=torch.float64, device=activity.device)
            if want_grouped:
                prob_sum = torch.zeros(grouped_est.n_states, dtype=torch.float64,
                                       device=activity.device)
                grouped_cond_sum = torch.zeros((), dtype=torch.float64, device=activity.device)
            n_have = 0
            while True:
                n = activity.shape[0]
                cond_sum += compute_response_conditional_entropy(activity).double() * n
                if want_grouped:
                    p_sum, c_sum = grouped_est.sufficient_statistics(activity)
                    prob_sum += p_sum.double()
                    grouped_cond_sum += c_sum.double()
                if want_lower or want_upper:
                    acts.append(loss_fn.grouping.probabilities(activity)
                                if native_grouped_kt else activity)
                if want_grouped_counting:
                    counter.update(activity, self._response_generator)
                if want_hard:
                    hard_codes.append((activity > 0.5).cpu())
                if want_counting:
                    a = activity.clamp(KT_EPS, 1.0 - KT_EPS)
                    sampled_codes.append(torch.bernoulli(
                        a, generator=self._response_generator).bool().cpu())
                n_have += n
                if n_have >= batch_size:
                    break
                E_k, concs_k, _ = env.sample_batch(min(chunk, batch_size - n_have),
                                                   receptor_indices=ri_batch)
                activity = self._activity(physics, E_k, concs_k, receptor_indices, env)

            h_cond = (cond_sum / n_have).item()
            stat["response_evaluation_samples"] = n_have

            if want_grouped:
                metrics = grouped_est.metrics_from_statistics(prob_sum, grouped_cond_sum, n_have)
                if "grouped_information" in requested:
                    stat.update({k: v.item() for k, v in metrics.items()})
                    stat.update(grouped_n_groups=grouped_est.n_groups,
                                grouped_n_states=grouped_est.n_states,
                                grouped_count_entropy_upper=math.log2(grouped_est.n_states))
                if native_grouped and "full_array_entropy" in requested:
                    stat["full_array_entropy"] = metrics["response_entropy_grouped"].item()
            if requested & {"conditional_entropy_response", "mutual_information_kt",
                            "mutual_information_kt_upper", "mutual_information_counting"}:
                stat["conditional_entropy_response"] = h_cond
            if want_hard:
                stat.update(codeword_entropy(torch.cat(hard_codes)))
            if want_counting:
                stat.update(response_counting_metrics(torch.cat(sampled_codes), h_cond))
            if want_grouped_counting:
                stat.update(counter.metrics())
            if want_lower or want_upper:
                big_soft = loss_fn.compute_soft_assignment(torch.cat(acts))
                mult = loss_fn.grouping.group_sizes if native_grouped_kt else None
                if want_lower:
                    i_lower = compute_kt_entropy(
                        big_soft, chunk_size=EVAL_TILE, return_mi=True,
                        use_compile=self.config.compile_kt, multiplicities=mult).item()
                    if native_grouped_kt and "full_array_entropy" in requested:
                        stat["full_array_entropy"] = i_lower + h_cond
                    if "entropy_kt" in requested:
                        stat["full_array_entropy_kt"] = i_lower + h_cond
                    if "mutual_information_kt" in requested:
                        stat["mutual_information_kt"] = i_lower
                if want_upper:
                    i_upper = compute_kt_upper_entropy(
                        big_soft, chunk_size=EVAL_TILE, return_mi=True,
                        multiplicities=mult).item()
                    if "entropy_kt_upper" in requested:
                        stat["full_array_entropy_kt_upper"] = i_upper + h_cond
                    if "mutual_information_kt_upper" in requested:
                        stat["mutual_information_kt_upper"] = i_upper
        return stat

    # -- training ----------------------------------------------------------

    def _train(self, env, physics, loss_fn, optimizer, receptor_indices):
        cfg = self.config
        is_mi = True        # every objective is an MI objective now
        start_temp = (compute_initial_temperature(env, receptor_indices)
                      if cfg.initial_temperature == "auto"
                      else float(cfg.initial_temperature))
        end_temp = cfg.temperature
        anneal_epochs = max(1, int(0.8 * cfg.epochs))
        ri_batch = receptor_indices if env.use_interface_model else None
        log_every = max(1, cfg.epochs // 100)

        stats = []
        for epoch in range(cfg.epochs):
            optimizer.zero_grad()
            frac = min(1.0, epoch / anneal_epochs)
            current_temp = (end_temp + (start_temp - end_temp) * (1.0 - frac)
                            if end_temp < start_temp else end_temp)
            physics.temperature = current_temp

            energies, concs, _ = env.sample_batch(cfg.batch_size, receptor_indices=ri_batch)
            activity = self._activity(physics, energies, concs, receptor_indices, env).float()
            loss = loss_fn(activity)
            loss.backward()
            optimizer.step()

            if epoch % log_every:
                continue
            if cfg.per_epoch_measure:
                # Measure at the SHARP endpoint, which is what the final number
                # uses; otherwise the curve describes a model nobody reports.
                physics.temperature = end_temp
                stat = self._eval_stats(env, physics, loss_fn, receptor_indices,
                                        cfg.test_batch_size)
                physics.temperature = current_temp
            else:
                metric = "train_mutual_information" if is_mi else "train_entropy"
                stat = {"loss": float(loss.item()), metric: float(-loss.item())}
            if is_mi:
                stat["train_mutual_information"] = float(-loss.item())
            stat["lr"] = optimizer.param_groups[0]["lr"]
            stats.append(stat)

        physics.temperature = end_temp
        return {k: [s[k] for s in stats] for k in stats[0]} if stats else {}

    def evaluate(self, measurements, batch_size, eval_chunk_size=None):
        """Measure the built model over batch_size inputs. One measurement, not repeated.

        Repeating would show evaluation NOISE, but a frequency-based entropy
        estimate is biased downward by an amount fixed by batch_size, so ten
        repeats of a small batch carry exactly the bias of one. At equal cost,
        put everything into batch_size.

        Values are stored as one-element lists so every consumer keeps one shape.
        """
        if self.env is None:
            raise RuntimeError("Call build() first, or load a checkpoint into it.")
        if eval_chunk_size is not None:
            self.config.eval_chunk_size = eval_chunk_size
        stat = self._eval_stats(self.env, self.physics, self.loss_fn,
                                self.receptor_indices, batch_size,
                                measurement_fns=tuple(measurements))
        return {k: [v] for k, v in stat.items()}


    def run(self):
        """Train and checkpoint. Deliberately does NOT measure.

        Measuring is a separate step because different estimators want
        different batch sizes: sampled counting needs a budget that grows with
        the number of cells, while KT costs O(B^2) and wants a modest one.
        Folding one of them into training would silently privilege it, and
        re-measuring would mean retraining. See tasks/_lib/measure.py.
        """
        # SweepLogger already wrote this, but a Runner built directly would
        # otherwise leave a directory nothing can load.
        self.logger.save_config(self.config)
        env, physics, _loss, optimizer, receptor_indices = self.build()
        history = self._train(env, physics, self.loss_fn, optimizer, receptor_indices)
        for i in range(len(next(iter(history.values()), []))):
            self.logger.save_stats(i, {k: v[i] for k, v in history.items()})
        self.logger.save_checkpoint(self.config.epochs, env, physics,
                                    receptor_indices, self.readout)
        return env
