# PORT STATUS — read this, act on it, then DELETE this file

Temporary handoff note. Not part of the project. Delete it once you have read it
and told the user where things stand.

Old repo: `/mnt/hcleroy/PostDoc2/octopus_smelling/opt_bin_resp` (frozen archive, untouched)
New repo: `/mnt/hcleroy/PostDoc2/octopus_smelling/opt_bin_resp_2` (this one, fresh git history)

## Where we stopped

Phases 0-3 are DONE and committed. Phase 4 (`analysis/`) is next and nothing of
it exists yet.

```
DONE   Phase 0   golden reference runs
DONE   Phase 1   model/     physics
DONE   Phase 2   estimate/  entropy, losses, measurements
DONE   Phase 3   train/     config, runner, sweep, io, batching
NEXT   Phase 4   analysis/  load.py + plot.py
       Phase 5   tasks/     5 frontier tasks
       Phase 6   ops/       manage_data.py, run.sh, docker, README
       Phase 7   tests      already partly done (11 passing)
```

8766 lines of `src/` became 3604. `pytest test/` is 11 passing.

## The rule that matters

**Every ported file was verified numerically against the old code, not just
read.** Both golden runs reproduce bit-identically. Keep doing this: write an
equivalence check in the scratchpad before moving on. It caught two real bugs
that reading had missed (`repertoire_weights` and `miller_madow_entropy` were
both called internally while appearing dead from outside).

## Do this next: Phase 4

The user chose "Generalise _load.py". The model to copy is
`opt_bin_resp/tasks/cells/fig2/analysis/_load.py` — read it first, it is already
close to the target and has good instincts (raises instead of printing, returns
a tidy frame, has a STYLE dict).

Build:

- `analysis/load.py`
  - `runs(goal, **filters) -> DataFrame`, one row per FINISHED run, config
    columns plus result columns. Build it on `train.io.iter_runs`.
  - `across_replicates(points, keys)` — mean plus min/max. Keep the fig2
    convention: replicates are independent worlds, so with only three of them a
    standard error would be fiction and min/max is the honest summary.
- `analysis/plot.py` — about 5 functions: `curves`, `points`, `band`,
  `umap_latent`, `umap_response`, plus a module-level `STYLE` dict.
  **No savefig anywhere, ever.** This is a standing user rule.

`analysis/` is the ONLY place matplotlib may be imported.
`test/test_imports.py` enforces that; do not weaken it.

## Decisions already made — do not reopen

| decision | what it means |
|---|---|
| frontier only | port 5 tasks: `analytic_baseline`, `large_R`, `two_families`, `fig2`, `fig3` |
| clean break on data | new code reads only runs it wrote. No compat shims. |
| `kt_mi` only | `annealed` dropped. The 3 receptor tasks need `entropy="kt_mi"` and a re-run. |
| silent, figure only | analysis prints nothing on the happy path and writes no files |
| explicit batch sizes | no `"auto"`. `train/batching.py` is an advisory tool; see `doc/batching.md` |
| mean cell readout only | threshold and noisy_or dropped |
| warm start dropped | runs in a sweep are independent, which is deliberate |
| port if cheap and physics | keep a capability the frontier does not use IF it is physics and small |

Diagnostics deliberately left behind in the old repo: `get_dose_response`,
`interaction_mu_interface`, `cell_ligand_responses`. The user said "neither for
now". Do not port them unless asked.

## Things that will bite you

1. **`Config.__post_init__` mutates.** It resolves `receptor_indices` and
   `cell_gene_sets` in place, so you cannot tell a user-supplied value from a
   derived one. That is why `sweep()` takes a **dict**, not a Config. Do not
   "improve" it into taking a Config.
2. **"Dead" symbols may have internal callers.** Grep excluding the definition
   line only, and count uses inside the same file.
3. **The 5 frontier tasks still say `entropy="annealed"` and `batch_size="auto"`.**
   Both are gone. When porting them in Phase 5, switch to `kt_mi` and compute
   explicit sizes with `python -m train.batching`.
4. **`block_size` is dead again** now that `annealed` is gone, but the 3 newest
   receptor tasks set it to 10. Delete that line when porting them.
5. The parent repo's `.gitmodules` has 3 dead entries and its gitlink is behind.
   Phase 6, not now.

## Open question for Phase 5

`kt_mi` is quadratic in batch size where `annealed` was not. Nobody has measured
what the switch costs `large_R` at R=64. If it turns out unaffordable, restoring
`annealed` is one file plus four functions in the old `bin_loss.py`. Flag this to
the user before porting `large_R`.

## Working agreement

The user wants to go step by step and be asked about anything genuinely
ambiguous. They have corrected the plan several times and those corrections were
right. Report honestly when a plan estimate turns out wrong — the config cut
landed at 48 fields, not the 32 the plan predicted, and saying so was better
than quietly missing the target.
