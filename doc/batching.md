# Choosing a batch size

Runs take explicit integers. Nothing is resolved at run time, so `config.json`
records the number that was actually used. `train/batching.py` computes a
suggestion; you paste it into the sweep script.

```bash
python -m train.batching --n_receptors 20 --n_ligands 100 --gpu_gb 80
```

These are **memory models, not measurements**. The safety factors were fitted on
an A100. If a run dies with an out-of-memory error, the model is wrong for your
case and the number needs lowering by hand.

Symbols: `B` batch size, `R` number of binary outputs the estimator sees, `L`
ligand pool size, `k` subunits per receptor (`k_sub`), `C` number of cells,
`R_pool` receptor pool size, `J` number of cell groups, `S` number of joint
count states.

---

## The three numbers

| field | used for | how often |
|---|---|---|
| `batch_size` | one training step | every step |
| `test_batch_size` | the per-epoch convergence curve | every logged epoch |
| `final_test_batch_size` | the single final measurement | once |

They differ because the costs differ. Training carries a backward graph, so it
is the most memory-hungry per sample. The per-epoch measurement runs under
`no_grad` but happens often, so it is kept at `4 * batch_size` to stop the
`O(B^2)` KT cost dominating every epoch. The final measurement runs once, so it
can be as large as memory allows.

---

## What binds, estimator by estimator

### KT (`kt`, `kt_mi`, `grouped_kt_mi`)

KT = Kolchinsky-Tracey. The estimator compares every pair of samples, so the
work is `O(B^2 * R)` and the tiles are `(m, m, R)`.

**Without `recompute_backward`, memory binds.** Autograd retains every tile for
the backward pass, so the retained graph is `B^2 * R * 4` bytes *regardless of
tile size*. Hence

```
B = sqrt( mem_budget / (R * 4 * SAFETY) ),    SAFETY = 3
```

`SAFETY = 3` covers the transient output of `torch.log` and its backward buffer,
which exist on top of the retained graph.

**With `recompute_backward`, compute binds.** Gradient checkpointing recomputes
each tile in backward instead of keeping it, dropping the graph to `O(B * R)`.
Memory stops being the limit and time becomes it, so the cap is a work budget:

```
B = sqrt( KT_COMPUTE_BUDGET / R ),   KT_COMPUTE_BUDGET = 2.1e10 pair-evaluations
```

Note this is a **time** budget, so it does not grow with a bigger GPU. Raise it
if you are willing to wait longer per step. For reference, the memory-bound work
on an 80 GiB A100 is about `5.3e9` pair-evaluations, so the default budget is 4x
that: roughly 2x the batch and 4x the step time.

The tile edge `KT_TRAIN_TILE = 4096` is a pure performance knob. Because the
retained graph is `B^2 * R` whatever the tile, the tile only sets a transient of
`m^2 * R` that rides underneath it. A larger tile means fewer, fatter kernel
launches, and KT is launch-bound. **It has no effect on the value**, which the
test suite checks.

### Exact Shannon (`shannon`)

Enumerates all `2^R` states, so `soft_assign` is `(B, 2^R)` float32 and the cap
is `mem_budget / (2^R * 4 * 4)`. Usable only below about R = 15. It exists as the
ground truth the KT bounds are checked against, not as a working objective.

### Exact grouped (`grouped_mi`)

Enumerates joint count states, retaining a `(B, S)` table rather than `(B, 2^C)`.
That is the whole point of grouping: `S = prod_j (n_j + 1)` instead of `2^C`.
Cap is `mem_budget / (S * 4 * 4)`, and `cell_grouped_max_states` refuses the run
outright when `S` is too large.

---

## Two caps that apply to everything

### Statistical saturation

Past a certain `B`, extra samples buy negligible variance reduction, so there is
no point paying for them:

- `shannon`: `100 * 2^R`, about a hundred samples per state.
- KT: `max(100 * 2^min(R,15), 16 * 2^(R/2))`. The second term is the one that
  bites at large R.
- grouped: `100 * S`.

This is a heuristic about sampling, **not** a guarantee that the estimate has
converged. Whether your MI estimate is sample-limited is a separate question,
answered by comparing it against `log2(B)` and by the
`missing_mass` / `K_frac` diagnostics in the counting measurements.

### Physics forward

The interface-model forward and backward hold several `(B, L, R_pool * k)`
float32 tensors simultaneously: `ab`, `dist_sq`, `exp(.)` and `E_open` inside
`_compute_energies`, then the open and closed log terms inside the receptor
activation, several retained for backward along with their gradients.

```
B = mem_budget / (L * R_pool * k * 4 * 16)
```

The factor 16 leaves headroom for the retained energy graph coexisting with the
KT tile during the loss backward. For the classic (non-interface) model the true
width is `n_genes` with no `k` axis, so charging `R_pool * k` is conservative.

**In cell mode these decouple.** The estimator sees `C` cells, but the physics
runs over the much larger receptor pool `R_pool`. Pass `--n_receptors C` and
`--n_physics_receptors R_pool`, or the physics cost is badly underestimated.

---

## Evaluation

Evaluation runs under `no_grad` and the all-pairs estimators tile internally, so
the binding term is the `(EVAL_TILE, B)` row buffer with `EVAL_TILE = 2048`:

```
B <= mem_budget / (EVAL_TILE * 4 * 4)
```

Samples beyond what one forward pass fits are generated in sub-batches, so this
is a pure memory bound rather than a limit on how many samples you may use. The
final measurement is additionally capped at the alphabet size, `2^R` or `S`,
since more samples than states is wasted.
