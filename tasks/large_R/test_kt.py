#!/usr/bin/env python3
"""large_R measurement: the Kolchinsky-Tracey bracket, as a cross-check on counting.

Counting cannot report more than log2(B) bits, so it understates a large array
however long you train it. KT costs O(B^2 * R), quadratic in the batch, so it
wants a modest one. They fail in opposite directions: where they agree, the
number is not an artefact of either, and where counting sits near its log2(B)
ceiling the KT bracket is the one to believe.

    python -m tasks.analytic_baseline.test_kt
"""
import argparse

from tasks._lib.measure import measure_goal

MEASUREMENTS = ("mutual_information_kt", "mutual_information_kt_upper")
LABEL = "kt"
BATCH_SIZE = 8192
EVAL_CHUNK = 2048

if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data_root", default="data/large_R")
    p.add_argument("--prefix", nargs="+", default=["shell_R"])
    p.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--device", default=None)
    a = p.parse_args()
    measure_goal(a.data_root, a.prefix, MEASUREMENTS, a.batch_size, LABEL,
                 eval_chunk_size=EVAL_CHUNK, device=a.device, overwrite=a.overwrite)
