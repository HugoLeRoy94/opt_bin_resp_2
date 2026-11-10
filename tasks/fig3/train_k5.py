#!/usr/bin/env python3
"""fig3 arm: k_sub = 5 subunits per receptor.

One of three arms; k_sub is the only thing that differs between them.

Trains and checkpoints only. Measure with tasks.fig3.test_counting and
tasks.fig3.test_kt.

    python -m tasks.fig3.train_k5 --dry_run
    python -m tasks.fig3.train_k5
"""
import argparse

from tasks.fig3._common import launch

if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data_root", default="data/fig3")
    p.add_argument("--dry_run", action="store_true")
    a = p.parse_args()
    launch(5, a.data_root, a.dry_run)
