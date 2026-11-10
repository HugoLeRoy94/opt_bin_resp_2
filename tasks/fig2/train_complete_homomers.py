#!/usr/bin/env python3
"""fig2 arm: COMPLETE-COVERAGE gene sets, HOMOMERS.

Every gene appears in at least one cell, but a cell builds only single-gene
receptors. The gene-set path cannot express this: expressing u and v
necessarily also builds every mixed receptor they allow, so the receptor list
is stated outright.

Writes to <data_root>/fig2_complete_homomers/, so the sweep directory names this file.

    python -m tasks.fig2.complete_homomers --dry_run
    python -m tasks.fig2.complete_homomers
"""
import argparse

from tasks.fig2._common import launch

if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data_root", default="data/fig2")
    p.add_argument("--dry_run", action="store_true",
                   help="print the design and the budget projection, run nothing")
    a = p.parse_args()
    launch("complete", "homomers", a.data_root, a.dry_run)
