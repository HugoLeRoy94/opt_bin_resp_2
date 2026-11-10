"""Advisory batch-size calculator. NOT called during a run.

Runs take explicit integers, so config.json records the number that was really
used instead of "auto" resolved at run time by a heuristic that may since have
changed. Use this to pick those integers once, then paste them in:

    python -m train.batching --n_receptors 20 --n_ligands 100 --gpu_gb 80

The reasoning behind every cap is in doc/batching.md. Read that before trusting
a number here: these are memory models, not measurements, and the safety
factors were fitted to an A100.

B = batch size, R = number of binary outputs, L = ligand pool, k = k_sub.
"""
import argparse
import math

# KT retains every (m, m, R) tile for backward, so the graph is B^2*R regardless
# of tile size. With recompute_backward the graph drops to O(B*R) and COMPUTE
# binds instead: this caps a step at that many Bhattacharyya pair-evaluations.
# It is a time budget, so it does not scale with GPU size.
KT_COMPUTE_BUDGET = 4 * 5.3e9

# Tile edge for KT during training. Only sets the transient (m^2*R) that rides
# under the retained B^2*R, so a bigger tile means fewer, fatter kernel launches
# for a little extra transient memory. No effect on values.
KT_TRAIN_TILE = 4096

# Tile edge during evaluation, where the binding term is the (tile, B) row
# buffer and B is large.
EVAL_TILE = 2048

B_MIN = 512


def suggest(n_receptors, mi="kt", grouped=False, n_ligands=1, k_sub=1, mem_budget_bytes=None,
            recompute_backward=False, n_physics_receptors=None,
            grouped_n_states=None, grouped_n_groups=None):
    """Return {batch_size, test_batch_size, measure_batch_size}.

    n_receptors is the width the ESTIMATOR sees: R in receptor mode, the number
    of cells C in cell mode. n_physics_receptors is the receptor POOL, which in
    cell mode is much larger and is what the physics forward actually costs.
    """
    if mem_budget_bytes is None:
        raise ValueError("Pass mem_budget_bytes: there is no sensible default, "
                         "and guessing one silently sizes for a GPU you may not have.")
    grouped_kt = grouped and mi == "kt"
    if grouped_kt and not (grouped_n_groups and grouped_n_states):
        raise ValueError("grouped_kt_mi needs grouped_n_groups and grouped_n_states.")
    kt_width = grouped_n_groups if grouped_kt else n_receptors
    kind = "grouped_mi" if (grouped and mi == "shannon") else mi

    if kind == "grouped_mi":
        if not grouped_n_states:
            raise ValueError("grouped_mi needs grouped_n_states.")
        # Exact enumeration retains a (B, S) table, not (B, 2^C).
        stats_cap = max(B_MIN, 100 * grouped_n_states)
        entropy_cap = max(1, mem_budget_bytes // (grouped_n_states * 4 * 4))
        physics = n_ligands * (n_physics_receptors or n_receptors) * k_sub * 4 * 16
        batch = min(stats_cap, entropy_cap, max(1, mem_budget_bytes // physics))
        return dict(batch_size=batch,
                    test_batch_size=min(4 * batch, stats_cap),
                    measure_batch_size=stats_cap)

    # Statistical saturation: beyond this, more samples buy negligible variance.
    if grouped_kt:
        stats_cap = max(B_MIN, 100 * grouped_n_states)
    elif kind == "shannon":
        stats_cap = max(B_MIN, 100 * (1 << n_receptors))
    else:                                        # kt
        stats_cap = max(B_MIN, max(100 * (1 << min(n_receptors, 15)),
                                   16 * int(2 ** (n_receptors / 2))))

    if kind == "shannon":
        # soft_assign is (B, 2^R) float32, 4x for backward.
        batch = min(stats_cap, max(B_MIN, mem_budget_bytes // ((1 << n_receptors) * 4 * 4)))
    else:
        if recompute_backward:
            cap = max(B_MIN, int(math.sqrt(KT_COMPUTE_BUDGET / kt_width)))
        else:
            SAFETY = 3                           # transient log output + backward
            cap = max(B_MIN, int(math.sqrt(mem_budget_bytes / (kt_width * 4 * SAFETY))))
        batch = min(stats_cap, cap)

    # Physics cap, all estimators. The interface forward+backward holds several
    # (B, L, R*k) float32 tensors at once; 16x leaves headroom for the retained
    # energy graph coexisting with the KT tile.
    per_sample = n_ligands * (n_physics_receptors or n_receptors) * k_sub * 4
    batch = min(batch, max(B_MIN, mem_budget_bytes // (per_sample * 16)))

    # Evaluation is no_grad and tiles internally, so the binding term is the
    # (EVAL_TILE, B) row buffer.
    eval_cap = max(B_MIN, mem_budget_bytes // (EVAL_TILE * 4 * 4))
    alphabet = grouped_n_states if grouped_kt else (1 << min(n_receptors, 62))
    final = max(batch, min(alphabet, eval_cap))
    return dict(batch_size=batch,
                test_batch_size=min(4 * batch, final),
                measure_batch_size=final)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n_receptors", type=int, required=True,
                   help="estimator width: R, or the cell count C in cell mode")
    p.add_argument("--mi", default="kt", choices=("kt", "shannon"))
    p.add_argument("--grouped", action="store_true")
    p.add_argument("--n_ligands", type=int, default=1)
    p.add_argument("--k_sub", type=int, default=5)
    p.add_argument("--gpu_gb", type=float, required=True, help="free GPU memory")
    p.add_argument("--recompute_backward", action="store_true")
    p.add_argument("--n_physics_receptors", type=int, default=None,
                   help="receptor pool size; cell mode only")
    p.add_argument("--grouped_n_states", type=int, default=None)
    p.add_argument("--grouped_n_groups", type=int, default=None)
    a = p.parse_args()
    out = suggest(a.n_receptors, a.mi, a.grouped, a.n_ligands, a.k_sub,
                  int(a.gpu_gb * (1 << 30)), a.recompute_backward,
                  a.n_physics_receptors, a.grouped_n_states, a.grouped_n_groups)
    width = a.grouped_n_groups or a.n_receptors
    print(f"\n  batch_size            = {out['batch_size']}")
    print(f"  test_batch_size       = {out['test_batch_size']}")
    print(f"  measure_batch_size    = {out['measure_batch_size']}"
          "   <- for a test script, not a config field")
    print(f"\n  KT pair-evaluations per step: {out['batch_size']**2 * width:.3g}")
    print("  See doc/batching.md for what each cap assumes.\n")


if __name__ == "__main__":
    main()
