"""One run's complete specification, and how a sweep is built from it.

Config is flat and all-scalar: it is exactly what gets written to config.json,
so a finished run is reproducible from its own directory. There is no second
"sweep config" dataclass duplicating these fields. A sweep is just a list of
Configs, produced by sweep().

__post_init__ resolves the two things that would otherwise be sampled at run
time: the receptor array and, in cell mode, the cell repertoires. Resolving
them here means the gene sets that actually got used are recorded, rather than
a seed that has to be replayed through the same code version to mean anything.

Cell mode is entered by setting any of cell_gene_sets, cell_receptors or
n_cells. In cell mode the receptor pool is derived from the cells.
"""
from dataclasses import asdict, dataclass, field, fields
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

from model.arrays import build_heteromer_array
from model.cells import build_cell_array

Number = Union[int, float]


@dataclass
class Config:
    # --- Chemical environment ---
    n_families: int
    n_ligands: int
    latent_dim: int
    family_spread: float
    average_family_distance: float
    distribution_type: str                     # "gaussian" or "uniform"
    observation_noise_sigma: float

    # --- Presence: hierarchical source -> ligand sampler ---
    n_presence_blocks: int
    mu_sources: float                          # mean active source blocks per sniff
    mu_ligands_per_source: float               # mean active ligands per source
    block_shared_conc_mean: bool

    # --- Concentration: log10(c) ~ Normal(conc_mean, conc_std) ---
    conc_mean: Union[float, Tuple[float, ...]]
    conc_std: Union[float, Tuple[float, ...]]

    # --- Receptor physics ---
    n_genes: int
    k_sub: int                                 # subunits per receptor
    temperature: float                         # width of the activation threshold
    kernel_params: Tuple[float, ...]           # (lambda,) Gaussian affinity width

    # --- Objective and training ---
    entropy: str                               # "kt_mi", "grouped_mi", "grouped_kt_mi", "kt", "shannon"
    batch_size: int                            # explicit; see doc/batching.md
    epochs: int
    lr: float
    test_batch_size: int                       # per-epoch measurement

    # --- Optional environment structure ---
    family_weights: Optional[Tuple[float, ...]] = None
    shared_unit_energies: bool = False
    fixed_max_energy: Optional[float] = None
    use_interface_model: bool = False

    # --- Schedules and performance (no effect on the model) ---
    initial_temperature: Union[float, str] = "auto"   # "auto" calibrates from a batch
    recompute_backward: bool = False           # gradient-checkpoint the KT inner loop
    compile_kt: bool = False
    per_epoch_measure: bool = True
    eval_chunk_size: Optional[int] = None

    # --- What to record DURING training, for the convergence curve ---
    # The final measurement is not here: it is a separate step, because
    # different estimators want different batch sizes. See tasks/_lib/measure.py.
    measurement_fns: Tuple[str, ...] = ("full_array_entropy",)

    # --- Receptor array. Give receptor_indices, or n_receptors to sample them. ---
    receptor_indices: Optional[List[List[int]]] = None
    n_receptors: Optional[int] = None
    receptor_sampling_strategy: str = "cascading"
    receptor_sampling_seed: Optional[int] = None

    # --- Cell mode ---
    cell_gene_sets: Optional[Sequence[Sequence[int]]] = None
    cell_receptors: Optional[Sequence[Sequence[Sequence[int]]]] = None
    n_cells: Optional[int] = None
    cell_sampling_strategy: str = "bernoulli"  # "bernoulli" or "size_pmf"
    cell_size_pmf: Optional[Sequence[float]] = None
    cell_max_genes: Optional[int] = None
    cell_sampling_seed: Optional[int] = None
    cell_stoichiometry: str = "multinomial"
    cell_pool_chunk: Optional[int] = None
    # Exact enumeration only. grouped_kt_mi / grouped_counting do not use it.
    cell_grouped_max_states: int = 65536

    # --- Labels. Descriptive only; nothing in model/, estimate/ or train/ ever
    # reads them. They record what a run MEANT, as opposed to what it did:
    # the design target a sweep asked for, which arm it belongs to. Config
    # records the gene sets a cell HAS, never the number that was requested,
    # and for a sampled strategy the two differ. analysis.load expands each key
    # into its own column.
    #
    # Prefer deriving a quantity from the config where you can. A tag that
    # disagrees with the config is a label error nothing will catch.
    tags: Dict[str, Any] = field(default_factory=dict)

    def is_cell_mode(self) -> bool:
        return (self.cell_gene_sets is not None or self.n_cells is not None
                or self.cell_receptors is not None)

    def __post_init__(self):
        wants_grouped = (
            self.entropy in ("grouped_mi", "grouped_kt_mi")
            or bool({"grouped_information", "grouped_counting"}
                    & set(self.measurement_fns or ())))
        if wants_grouped and not self.is_cell_mode():
            raise ValueError("Grouped losses and measurements require cell mode.")

        if self.is_cell_mode():
            cells = build_cell_array(
                repertoires=self.cell_receptors, k_sub=self.k_sub,
                n_genes=self.n_genes, gene_sets=self.cell_gene_sets,
                n_cells=self.n_cells, strategy=self.cell_sampling_strategy,
                size_pmf=self.cell_size_pmf, max_genes=self.cell_max_genes,
                seed=self.cell_sampling_seed, stoichiometry=self.cell_stoichiometry,
                use_interface_model=self.use_interface_model)
            # Persist what the cells TURNED OUT to be, derived or not.
            self.cell_gene_sets = [list(gs) for gs in cells.gene_sets]
            self.n_cells = cells.n_cells
            self.receptor_indices = cells.receptor_indices.tolist()
            return

        if self.receptor_indices is None:
            if self.n_receptors is not None:
                self.receptor_indices = build_heteromer_array(
                    self.n_genes, self.k_sub, self.n_receptors,
                    strategy=self.receptor_sampling_strategy,
                    seed=self.receptor_sampling_seed,
                    use_interface_model=self.use_interface_model).tolist()
            else:
                # No array asked for: one homomer per gene.
                self.receptor_indices = [[i] * self.k_sub for i in range(self.n_genes)]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Config":
        return cls(**d)

    def __str__(self) -> str:
        lines = ["=== Config ==="]
        for key, value in asdict(self).items():
            shown = f"<{len(value)} entries>" if isinstance(value, list) and len(value) > 8 else value
            lines.append(f"  {key:30s} {shown}")
        return "\n".join(lines)


def sweep(base: Dict[str, Any], **axes) -> List[Config]:
    """One Config per sweep step. Axes are ZIPPED, never crossed.

    `base` is a plain dict of Config arguments, not a Config: Config.__post_init__
    resolves receptor_indices and cell_gene_sets, and a resolved array must never
    be inherited by a step that changed n_genes. Building each step from the raw
    spec makes that impossible.

    Zipping means a sweep is a list of design points you chose, not a grid you got
    by accident:

        sweep(BASE, n_genes=[5, 10, 15], conc_std=[1.2, 1.4, 1.6])

    is three runs, not nine. All axes must share one length. Pass a list of tuples
    for a tuple-valued field.
    """
    if not axes:
        return [Config(**base)]
    lengths = {name: len(values) for name, values in axes.items()}
    if len(set(lengths.values())) != 1:
        raise ValueError("All sweep axes must share one length, got "
                         + ", ".join(f"{k}={n}" for k, n in sorted(lengths.items())))
    return [Config(**{**base, **{k: v[i] for k, v in axes.items()}})
            for i in range(next(iter(lengths.values())))]
