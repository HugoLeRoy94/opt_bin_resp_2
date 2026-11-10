# octopus array

Optimizing the mutual information between a combinatorial receptor array and a
chemical latent space. MI = mutual information, in bits.

## Layout

```
model/      physics: latent space, receptors, array composition, cells
estimate/   entropy and MI estimators, objectives, measurements
train/      config, training loop, sweeps, disk IO
analysis/   loading results and drawing them. The ONLY matplotlib
tasks/      one directory per scientific question
test/       pytest
```

The import direction is one-way and `test/test_imports.py` enforces it:
`model` -> `estimate` -> `train` -> `analysis`. Nothing outside `tasks/`
imports `analysis`, and nothing in `model`, `estimate` or `train` imports
matplotlib. A training container never needs it.

## Install, once

```bash
cd opt_bin_resp_2
pip install -e . --no-deps
```

This is what makes `import analysis` work from any directory: a script, a
notebook, an IDE interactive cell. Without it the imports only resolve when
the working directory happens to be the repository root.

It has to be an install rather than a path line in each file. An interactive
cell sets the working directory to the workspace root rather than the file's
folder, and `__file__` is undefined there, so neither can locate the
repository. `--no-deps` stops pip touching your torch or numpy.

Editable, so edits take effect with no reinstall. Re-run it only if you move
the repository or add a top-level package.

## Running

From anywhere, once installed.

```bash
python -m tasks.<task>.train_<arm>  --dry_run     # print the design, run nothing
python -m tasks.<task>.train_<arm>                # train, write checkpoints
python -m tasks.<task>.test_<label>               # measure the checkpoints
python tasks/<task>/figure.py                     # draw
```

Training and measuring are separate steps. Training writes `config.json`,
`stats.csv` and `best_model.pt` and nothing else. A test script loads those
checkpoints and writes `test_<label>.json`.

They are separate because the estimators want incompatible budgets. Sampled
counting needs a batch far above the effective alphabet, which grows with the
array; KT costs `O(B^2 R)` and wants a modest one. They fail in opposite
directions, so running both is a cross-check, and re-measuring never means
retraining.

`--dry_run` costs nothing and prints the grid, the batch sizes and, where a
task has one, what the closed form expects. Use it first.

## The six tasks

| task | question | arms |
|---|---|---|
| `fig1` | does splitting a gene budget into more receptors buy information? | `train_heteromers` |
| `fig2` | does information keep rising as a cell expresses more genes? | 4: {complete, bernoulli} x {heteromers, homomers} |
| `fig3` | can the MI maximum be moved by changing `k_sub`? | `train_k2`, `train_k3`, `train_k5` |
| `analytic_baseline` | does the closed form hold for `R <= D`? | `train_sweep_R` |
| `large_R` | what happens past `R = D`, where counting takes over? | `train_sweep` |
| `two_families` | does an array facing two distant families split in two? | `train_sweep` |

`two_families` needs `large_R` run first at `D = 2`, `conc_std = 1.7372`: it
takes `G(n)`, the single-family MI, from there rather than measuring it.

## Batch sizes are explicit

No `"auto"`. A run records the number it actually used, rather than one a
heuristic produced at run time and may produce differently later.

```bash
python -m train.batching --n_receptors 20 --n_ligands 100 --gpu_gb 80
```

suggests the three sizes. `doc/batching.md` explains every cap and the safety
factors, which were fitted on an A100. They are memory models, not
measurements: if a run dies with an out-of-memory error, lower the number.

## Two things the code will refuse to do

**Train on entropy.** Every objective maximizes a bound on `I(Y;X)`. Maximizing
`H(Y)` alone has a degenerate optimum: an array of fair coins scores the
maximum and carries no information. `mi` selects the estimator (`kt` or
`shannon`) and `grouped` selects whether identical cells collapse to counts.

**Write a file from an analysis.** No figure is ever saved and nothing is
printed on the happy path. Analyses raise instead, when pooling runs that
should not be pooled or when a measurement sits at its sampling ceiling.
