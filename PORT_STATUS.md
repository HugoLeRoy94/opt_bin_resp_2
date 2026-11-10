# PORT STATUS — read this, act on it, then DELETE this file

Temporary handoff note. Not part of the project.

Old repo: `/mnt/hcleroy/PostDoc2/octopus_smelling/opt_bin_resp` (frozen archive)
New repo: this one.

## Where we stopped

All six frontier tasks are ported. The user is taking it to the GPU and will
report problems. **Expect the next message to be a bug report from a real run.**

```
DONE   Phases 0-4   model/ estimate/ train/ analysis/
DONE   Phase 5      fig1, fig2, fig3, analytic_baseline, large_R, two_families
LEFT   Phase 6      ops/: manage_data.py curate+sync, run_remote.sh, docker,
                    .gitmodules cleanup. README.md is written.
LEFT                analytic_baseline geometry figures (geometry_norms,
                    geometry_orthogonality), deferred by the user. They need
                    _lib/runs.py; only its two energy accessors are ported so
                    far, as tasks/_lib/energies.py.
```

7827 lines. `pytest test/` is 138 passing, 2 skipped.

## NOTHING HAS TOUCHED A GPU

Every smoke test was CPU with 6-8 epochs and tiny batches. The pipelines are
verified; no number produced so far means anything scientifically. When the
user reports a problem, the first question is whether it is a port bug or a
scale effect that CPU smoke tests could not show. Likely candidates:

- batch sizes from `train/batching.py` are memory MODELS fitted on an A100,
  not measurements. An OOM means the model is wrong for that case, not that
  the port is broken. Lower the number and say so.
- `compile_kt=True` (fig1, analytic_baseline) has never been exercised.
- `recompute_backward=True` (fig2, fig3) has never been exercised at scale.
- the KT tile and eval tile constants in `train/runner.py`.

## The rule that matters

**Verify numerically against the old code, not by reading.** It has caught, so
far: two symbols that looked dead but had internal callers, two sweeps merging
into one directory, a loader crash on unset fields, a mislabelled point in the
user's existing fig1 data, a Runner that never wrote its config, a measurement
reporting no sample count, and a figure column silently overwritten by
bookkeeping. Every one came from RUNNING something, not reading it.

## Decisions — do not reopen

| decision | what it means |
|---|---|
| clean break on data | new code reads only runs it wrote |
| objective is `mi` x `grouped` | no objective maximizes H(Y); that optimum is fair coins |
| train and measure are separate | training writes a checkpoint and stops |
| a run is finished when checkpointed | not when measured |
| silent, figure only | analysis prints nothing on the happy path, writes no files |
| explicit batch sizes | no `"auto"`; `train/batching.py` is advisory |
| std, not min/max | `across_replicates` defaults to std; minmax removed |
| tags on Config | replaces experiment.json; descriptive only |
| one file per arm | `train_<arm>.py`, `test_<label>.py`, `figure.py` |
| mean cell readout only | threshold and noisy_or dropped |
| warm start dropped | runs in a sweep are independent, deliberately |

## Things that will bite you

1. **`Config.__post_init__` mutates.** That is why `sweep()` takes a dict.
2. **"Dead" symbols may have internal callers.** Count uses inside the file.
3. **Group by the design TARGET, plot the realised value.** Under a sampled
   gene-set strategy the realised mean drifts further than the target spacing.
4. **The receptor pool saturates.** `pool_size()`; asking for more raises.
5. **`two_families` needs `large_R` first**, at D=2, conc_std=1.7372.
6. **Two passes reporting one key** keep both as `"<key> (<label>)"` and drop
   the plain name. H(Y|X) at two budgets is the normal case.

## Working agreement

Step by step; ask when genuinely ambiguous. The user has corrected the plan
repeatedly and was right every time, including on the statistics and on the
objective. Avoid jargon: "arm", "drift", "n", "self-test" all needed
explaining, and the explanations were worth more than the shorthand. Define a
term the first time it is used. When a previous claim turns out wrong, say so
plainly and correct it: that has happened twice (the annealed cost, the
self-test wording) and both corrections mattered.
