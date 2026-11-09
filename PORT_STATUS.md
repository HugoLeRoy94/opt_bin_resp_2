# PORT STATUS — read this, act on it, then DELETE this file

Temporary handoff note. Not part of the project. Delete it once you have read it
and told the user where things stand.

Old repo: `/mnt/hcleroy/PostDoc2/octopus_smelling/opt_bin_resp` (frozen archive, untouched)
New repo: `/mnt/hcleroy/PostDoc2/octopus_smelling/opt_bin_resp_2` (this one)

## Where we stopped

```
DONE   Phase 0   golden reference runs
DONE   Phase 1   model/     physics
DONE   Phase 2   estimate/  entropy, losses, measurements
DONE   Phase 3   train/     config, runner, sweep, io, batching
DONE   Phase 4   analysis/  load, plot, umap
       Phase 5   tasks/     fig1 DONE, fig2 DONE
                            NEXT: fig3, then large_R, two_families, analytic_baseline
       Phase 6   ops/       manage_data.py, run.sh, docker, README
       Phase 7   tests      73 passing
```

5199 lines total. `pytest test/` is 73 passing.

## Do this next: fig3

`opt_bin_resp/tasks/cells/fig3/` — three arms, k_sub = 2, 3, 5, asking whether the
MI maximum moves when receptors have fewer subunits. It reuses `tasks/_lib/`
(already ported) and is structurally the same as fig2, so copy that pattern:

- one file per arm named for its data directory (`k2.py`, `k3.py`, `k5.py`)
- `tasks/fig3/_common.py` for the grid, the world draw and the config
- `WORLDS = 1` at the top, the one number to raise for replicates
- tag `target_genes` and the arm; group by the target, plot `genes_per_cell`
- run `counting_budget_report` before launching

fig3 uses `complete_coverage_gene_sets` (the integer version), not `_mixed`.

## The rule that matters

**Verify numerically against the old code, not by reading.** Every ported file
was checked this way. It has caught, so far: two symbols that looked dead but
had internal callers, two sweeps merging into one directory, a loader crash on
unset optional fields, and a mislabelled point in the user's existing fig1 data.
Write the check in the scratchpad before moving on.

Running a tiny smoke sweep end to end is how most of those were found. Do it.

## Decisions already made — do not reopen

| decision | what it means |
|---|---|
| frontier only | fig1, fig2, fig3, large_R, two_families, analytic_baseline |
| clean break on data | new code reads only runs it wrote. No compat shims. |
| `kt_mi` / `kt` only | `annealed` dropped; the 3 receptor tasks need `entropy="kt_mi"` and a re-run |
| silent, figure only | analysis prints nothing on the happy path and writes no files, ever |
| explicit batch sizes | no `"auto"`; `train/batching.py` is advisory, see `doc/batching.md` |
| mean cell readout only | threshold and noisy_or dropped |
| warm start dropped | runs in a sweep are independent, deliberately |
| std, not min/max | `across_replicates` defaults to std; minmax removed, its width grows with n |
| tags on Config | replaces `experiment.json`; descriptive only, never read by the model |
| one file per arm | sweep_name == filename == data dir, so a figure traces back to its script |
| port if cheap and physics | keep a capability the frontier does not use IF it is physics and small |

Diagnostics deliberately left in the old repo: `get_dose_response`,
`interaction_mu_interface`, `cell_ligand_responses`, and the response-UMAP
panels that depend on them. The latent UMAP panel IS ported and needs none of
them. Do not port the rest unless asked.

## Things that will bite you

1. **`Config.__post_init__` mutates**, resolving `receptor_indices` and
   `cell_gene_sets` in place. That is why `sweep()` takes a **dict**, not a
   Config. Do not "improve" it.
2. **"Dead" symbols may have internal callers.** Grep excluding only the
   definition line, and count uses inside the same file.
3. **Group by the design TARGET, plot the realised value.** Under a sampled
   gene-set strategy the realised mean drifts from the target by more than the
   spacing between targets, so grouping by the realised value gives every world
   its own x and n = 1 everywhere. This is what tags exist for.
4. **large_R, two_families and analytic_baseline still say `entropy="annealed"`
   and `batch_size="auto"`.** Both are gone. Switch to `kt_mi` and compute
   explicit sizes with `python -m train.batching`. Also delete their
   `block_size = 10` line, which died with annealed.
5. **The receptor pool saturates.** `pool_size(n_genes, k_sub, interface)` says
   how many distinct receptors exist; asking for more now raises. 2 genes with
   k_sub=5 admit only 8.
6. The parent repo's `.gitmodules` has 3 dead entries and its gitlink is
   behind. Phase 6, not now.

## Open question for large_R

`kt_mi` is quadratic in batch size where `annealed` was not, and nobody has
measured what the switch costs at R = 64. If it proves unaffordable, restoring
`annealed` is one file plus four functions in the old `bin_loss.py`. Flag this
to the user BEFORE porting large_R.

## Working agreement

Step by step; ask when something is genuinely ambiguous. The user has corrected
the plan several times and was right each time, including on the statistics.
Avoid jargon: "arm", "drift" and "n" all needed explaining, and the explanations
were worth more than the shorthand. Define a term the first time it is used.
Report honestly when an estimate turns out wrong.
