#!/usr/bin/env python3
"""fig1 measurement: the Kolchinsky-Tracey bracket on every trained run.

Records the certified LOWER and UPPER bounds on I(Y;X), the information the
array carries about the stimulus. The lower bound is what training maximised
and is the number the figure reports; the upper bound says how tight that
claim is.

It reports MI rather than H(Y) deliberately. H(Y) counts response noise as
signal, and at zero observation noise the two nearly coincide, but only one of
them is the quantity the figure is about.

KT costs O(B^2 * R), quadratic in the batch, so the budget here is modest and
has nothing to do with the training batch. Run after training:

    python -m tasks.fig1.test_kt --dry_run
    python -m tasks.fig1.test_kt
"""
import argparse

from tasks._lib.measure import measure_goal

MEASUREMENTS = ("mutual_information_kt", "mutual_information_kt_upper")
LABEL = "kt"
BATCH_SIZE = 8192          # KT is O(B^2 R); see doc/batching.md
EVAL_CHUNK = 2048

if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data_root", default="data/fig1")
    p.add_argument("--prefix", nargs="+", default=["ng"],
                   help="sweep name prefixes to measure")
    p.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    p.add_argument("--overwrite", action="store_true",
                   help="re-measure runs that already have test_kt.json")
    p.add_argument("--device", default=None)
    a = p.parse_args()
    measure_goal(a.data_root, a.prefix, MEASUREMENTS, a.batch_size, LABEL,
                 eval_chunk_size=EVAL_CHUNK, device=a.device, overwrite=a.overwrite)
