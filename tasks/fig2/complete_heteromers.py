#!/usr/bin/env python3
"""fig2 arm: COMPLETE-COVERAGE gene sets, HETEROMERS.

Every gene appears in at least one cell, and each cell assembles every receptor
its genes allow, homomers and heteromers alike, weighted by multinomial
abundance.

Writes to <data_root>/fig2_complete_heteromers/, so the sweep directory names this file.

    python -m tasks.fig2.complete_heteromers --dry_run
    python -m tasks.fig2.complete_heteromers
"""
import argparse

from tasks.fig2._common import launch

if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data_root", default="data/fig2")
    p.add_argument("--dry_run", action="store_true",
                   help="print the design and the budget projection, run nothing")
    a = p.parse_args()
    launch("complete", "heteromers", a.data_root, a.dry_run)
