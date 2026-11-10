#!/usr/bin/env python3
"""fig3 measurement: sampled grouped counting.

Counts integer group counts drawn from the trained cells and estimates MI,
H(Y), H(Y|X) and alphabet coverage as ONE estimator family, so
MI = H(Y) - H(Y|X) holds exactly rather than approximately.

The budget is large because sampled counting cannot report more than log2(B)
bits while H(Y|X) is computed analytically with no such cap. When the two meet,
the reported MI collapses and then goes negative. The required B grows with the
number of cells, which is why C = 50 was dropped from the design. See
tasks/_lib/budget.py, and check grouped_counting_missing_mass in the figure.

    python -m tasks.fig3.test_counting
"""
import argparse

from tasks._lib.measure import measure_goal

MEASUREMENTS = ("grouped_counting",)
LABEL = "counting"
BATCH_SIZE = 2 ** 26
EVAL_CHUNK = 512

if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data_root", default="data/fig3")
    p.add_argument("--prefix", nargs="+", default=["fig3_k"])
    p.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--device", default=None)
    a = p.parse_args()
    measure_goal(a.data_root, a.prefix, MEASUREMENTS, a.batch_size, LABEL,
                 eval_chunk_size=EVAL_CHUNK, device=a.device, overwrite=a.overwrite)
