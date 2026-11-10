#!/usr/bin/env python3
"""fig3 arm: k_sub = 2 subunits per receptor.

The pure-dilution control. At G = 5 the pool is exhausted at 2 genes per
cell, so everything past g = 2 is dilution with zero pool gain behind it.

Trains and checkpoints only. Measure with tasks.fig3.test_counting and
tasks.fig3.test_kt.

    python -m tasks.fig3.train_k2 --dry_run
    python -m tasks.fig3.train_k2
"""
import argparse

from tasks.fig3._common import launch

if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data_root", default="data/fig3")
    p.add_argument("--dry_run", action="store_true")
    a = p.parse_args()
    launch(2, a.data_root, a.dry_run)
