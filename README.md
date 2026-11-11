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

`two_families` needs `analytic_baseline` run first at `D = 8`, `conc_std = 0.2171`:
it takes `G(n)`, the single-family MI, from there rather than measuring it.

## Batch sizes are explicit

No `"auto"`. A run records the number it actually used, rather than one a
heuristic produced at run time and may produce differently later.

```bash
python -m train.batching --n_receptors 20 --n_ligands 100 --gpu_gb 80
```

suggests the three sizes. `doc/batching.md` explains every cap and the safety
factors, which were fitted on an A100. They are memory models, not
measurements: if a run dies with an out-of-memory error, lower the number.

## On the cluster, once

The cluster has no direct outbound HTTPS, so teach git about the university
proxy first. In `git config`, not the shell environment: `run_remote.sh` runs
git over ssh, and exported variables in `.bashrc` or `.profile` are not read by
every kind of shell.

```bash
ssh leroy@10.187.172.7 'git config --global http.proxy  http://proxy.unige.it:8080/'
ssh leroy@10.187.172.7 'git config --global https.proxy http://proxy.unige.it:8080/'
```

The launcher pulls into an existing checkout; it does not create one.

```bash
ssh leroy@10.187.172.7 'git clone https://github.com/HugoLeRoy94/opt_bin_resp_2.git /home/leroy/opt_bin_resp_2'
```

If that asks for credentials, store them once so later pulls are
non-interactive. A prompt inside `run_remote.sh` would never be seen, because
the commands run over ssh with no terminal attached:

```bash
ssh -t leroy@10.187.172.7 'cd /home/leroy/opt_bin_resp_2 && git config credential.helper store && git pull'
```

No install step and no rebuild: the container mounts the source and sets
`PYTHONPATH=/app`, and `requirements.txt` is a subset of what the existing
image already has. Override the paths with `OCTOPUS_REMOTE_ROOT`,
`OCTOPUS_REMOTE_DATA_ROOT` and `OCTOPUS_DATA_SERVER` if they differ.

## On the cluster

```bash
ops/run_remote.sh <task> <script> [gpu] [-- args]

ops/run_remote.sh analytic_baseline train_sweep_R          # GPU 0
ops/run_remote.sh analytic_baseline train_sweep_R 2 -- --R 2 4 8
ops/run_remote.sh analytic_baseline test_counting 0
```

It pulls, then launches the task in a detached tmux session with a GPU memory
monitor beside it, and prints how to attach, list and kill. When the run ends
the GPU trace and the stdout log are moved into the sweep directory, so a later
sync brings them back with the runs.

Kill one run with `tmux kill-session`, never `tmux kill-server`: that kills
every session on a shared machine.

**The container is the existing one.** `requirements.txt` is a strict subset of
what the `opt_bin_resp` image already installed, and `PYTHONPATH=/app` is set in
the compose file rather than baked into the image, so nothing needs rebuilding.
If `docker images` shows a different tag:

```bash
OCTOPUS_IMAGE=<tag> ops/run_remote.sh ...
```

Scripts run as `python3 -m tasks.<task>.<script>` from `/app`, which is what
puts the packages on the path. Running the file by its path would not.

## Results

```bash
python ops/manage_data.py curate [goal ...]     # label sweeps keep or delete
python ops/manage_data.py sync   [goal ...]     # apply deletions, mirror down
```

Raw data live on the cluster under `/storage/leroy/data_2`, separate from the
old repo's `/storage/leroy/data`, so the two code bases never read each
other's runs. Locally they mirror to this repo's `data/`.

The cluster is authoritative for raw data; `curation.csv` is authoritative for
the human keep/delete decision, and is tracked in git so it survives the data
being deleted or re-synced. A kept sweep needs a short label saying why.

`sync` mirrors with `--delete`, so a sweep that exists only locally disappears.
It names those first and asks before removing anything.

`analysis.load.runs()` surfaces the decision as a `curation` column, so a figure
can restrict itself to kept sweeps. It does not filter on it by itself.

## Two things the code will refuse to do

**Train on entropy.** Every objective maximizes a bound on `I(Y;X)`. Maximizing
`H(Y)` alone has a degenerate optimum: an array of fair coins scores the
maximum and carries no information. `mi` selects the estimator (`kt` or
`shannon`) and `grouped` selects whether identical cells collapse to counts.

**Write a file from an analysis.** No figure is ever saved and nothing is
printed on the happy path. Analyses raise instead, when pooling runs that
should not be pooled or when a measurement sits at its sampling ceiling.
