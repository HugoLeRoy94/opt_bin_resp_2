#!/usr/bin/env python3
"""fig2 arm: BERNOULLI gene sets, HOMOMERS.

Each gene is expressed independently at probability p, and a cell builds only
single-gene receptors. The two independent choices of fig2, crossed.

Writes to <data_root>/fig2_bernoulli_homomers/, so the sweep directory names this file.

    python -m tasks.fig2.bernoulli_homomers --dry_run
    python -m tasks.fig2.bernoulli_homomers
"""
import argparse

from tasks.fig2._common import launch

if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data_root", default="data/fig2")
    p.add_argument("--dry_run", action="store_true",
                   help="print the design and the budget projection, run nothing")
    a = p.parse_args()
    launch("bernoulli", "homomers", a.data_root, a.dry_run)
