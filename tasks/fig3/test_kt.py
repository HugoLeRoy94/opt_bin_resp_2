#!/usr/bin/env python3
"""fig3 measurement: the Kolchinsky-Tracey bracket, as a cross-check on counting.

The two estimators fail in OPPOSITE directions, which is the point of running
both. Sampled counting needs a budget above the effective alphabet 2^H(K),
which grows with the number of cells, so it degrades as C rises. KT costs
O(B^2 * C), only linear in C but quadratic in the batch, so it degrades as B
rises. Where they agree, the number is not an artefact of either.

A modest batch here is deliberate, and is not the counting budget.

    python -m tasks.fig3.test_kt
"""
import argparse

from tasks._lib.measure import measure_goal

MEASUREMENTS = ("mutual_information_kt", "mutual_information_kt_upper")
LABEL = "kt"
BATCH_SIZE = 8192
EVAL_CHUNK = 2048

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
