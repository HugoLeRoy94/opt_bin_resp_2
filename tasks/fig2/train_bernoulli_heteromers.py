#!/usr/bin/env python3
"""fig2 arm: BERNOULLI gene sets, HETEROMERS.

Each gene is expressed independently at probability p, so the number per cell
is induced rather than chosen and a gene can end up expressed nowhere. Each
cell assembles every receptor its genes allow.

Writes to <data_root>/fig2_bernoulli_heteromers/, so the sweep directory names this file.

    python -m tasks.fig2.bernoulli_heteromers --dry_run
    python -m tasks.fig2.bernoulli_heteromers
"""
import argparse

from tasks.fig2._common import launch

if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data_root", default="data/fig2")
    p.add_argument("--dry_run", action="store_true",
                   help="print the design and the budget projection, run nothing")
    a = p.parse_args()
    launch("bernoulli", "heteromers", a.data_root, a.dry_run)
