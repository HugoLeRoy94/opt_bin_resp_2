#!/usr/bin/env python3
"""large_R measurement: hard codewords, the quantity the theory predicts.

Observation noise is zero and the threshold is near-hard, so the response is a
deterministic function of the stimulus: H(Y|X) is about 0 and the codeword
entropy IS the mutual information. That is what codeword_entropy estimates,
by binarising at 0.5, counting distinct R-bit patterns and applying the
Miller-Madow correction.

Read codeword_entropy_missing_mass and codeword_entropy_log2B with it. The
plug-in estimate cannot exceed log2(B), so a point near that ceiling is
reporting the budget rather than the array.

    python -m tasks.analytic_baseline.test_counting
"""
import argparse

from tasks._lib.measure import measure_goal

MEASUREMENTS = ("codeword_entropy", "mutual_information_counting")
LABEL = "counting"
BATCH_SIZE = 2 ** 18        # log2 B = 18, the ceiling on any counting estimate
EVAL_CHUNK = 4096

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
