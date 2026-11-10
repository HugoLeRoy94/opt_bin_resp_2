"""The two golden configurations, in fields that survive into the new Config.

Shared by make_golden.py (runs them against the OLD repo) and test_golden.py
(runs them against the new one). Only fields that exist in BOTH live here; the
generator adds the pinned fields the new Config drops.

Kept small enough to run on CPU in about a minute each. SEED is an engineering
determinism knob so the two codebases see identical draws. It is not a
scientific control and no result is ever reported from these runs.
"""

SEED = 20261106

RECEPTOR = dict(
    n_families=3, n_ligands=12, latent_dim=3,
    family_spread=0.3, average_family_distance=1.0,
    distribution_type="gaussian", observation_noise_sigma=0.0,
    n_presence_blocks=1, mu_sources=1.0, mu_ligands_per_source=2.0,
    block_shared_conc_mean=False, conc_mean=0.0, conc_std=1.0,
    n_genes=4, k_sub=5, temperature=1.0,
    kernel_params=(1.0,),
    batch_size=64, entropy="kt_mi", epochs=20, lr=0.05,
    test_batch_size=64, initial_temperature=1.0,
    n_receptors=6, receptor_sampling_strategy="cascading",
    receptor_sampling_seed=7,
    per_epoch_measure=False,
    measurement_fns=("mutual_information_kt",),
)

# What the separate measuring step records. Training no longer measures, so the
# golden is reproduced by run() followed by evaluate().
RECEPTOR_MEASURE = dict(
    measurements=("mutual_information_kt", "full_array_entropy",
                  "conditional_entropy_response"),
    batch_size=128,
)

CELL = dict(
    n_families=3, n_ligands=12, latent_dim=3,
    family_spread=0.3, average_family_distance=1.0,
    distribution_type="gaussian", observation_noise_sigma=0.0,
    n_presence_blocks=1, mu_sources=1.0, mu_ligands_per_source=2.0,
    block_shared_conc_mean=False, conc_mean=0.0, conc_std=1.0,
    n_genes=4, k_sub=5, temperature=1.0,
    kernel_params=(1.0,),
    batch_size=64, entropy="grouped_kt_mi", epochs=20, lr=0.05,
    test_batch_size=64, initial_temperature=1.0,
    cell_gene_sets=((0, 1), (1, 2), (2, 3), (0, 3), (0, 2)),
    per_epoch_measure=False,
    measurement_fns=("grouped_information",),
)

CELL_MEASURE = dict(
    measurements=("grouped_information", "grouped_counting"),
    batch_size=128,
)

# Fields the new Config drops because every call site pins them to one value.
# The old SingleRunConfig requires them explicitly.
PINNED = dict(environment_geometry="asymmetric", conc_model_type="lognormal",
              use_scheduler=False, affinity_kernel="gaussian",
              cell_readout="mean", cell_recalibrate_every=0)
