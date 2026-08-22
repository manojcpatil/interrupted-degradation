"""
code_09_audit.py  ->  the exhaustive checks behind Section 5
============================================================
Correctness evidence for the embeddings. Two checks, both of which compare the
chain against the definition rather than against another chain.

A. Exhaustive path audit.  For every sequence in {N,W,C}^n the count produced
   by replaying the embedded chain is compared with the count produced by the
   definition-based scanner of Algorithm 1.  The two are independent: the
   scanner never forms A or B.

B. Brute-force PMF.  All 3^n paths are enumerated and weighted by their
   probabilities; the resulting exact mass function is compared with the one
   obtained from the matrix recursion.

No table is written. The manuscript states both results in the prose of
Section 5; the numbers behind them are left in Output/ so that the claims can
be checked without rerunning the whole pipeline.

Outputs: Output/audit_paths.csv, Output/audit_pmf.csv

License: MIT.
"""

from __future__ import annotations

import argparse
import itertools

import numpy as np

import mcem_core as C
from common import banner, save_environment, write_csv

GRID = [(2, 1), (3, 1), (3, 2), (4, 2)]


def audit_paths(grid=GRID):
    banner("A. Exhaustive path audit against Definition 1")
    rows = []
    print("  %-4s %-3s %-3s %-4s %10s | %s"
          % ("fam", "k", "r", "n", "paths", "chain versus definition"))
    for k, r in grid:
        n = 9 if k <= 3 else 10
        paths = list(itertools.product((0, 1, 2), repeat=n))
        for fam in C.FAMILIES:
            bad = 0
            for seq in paths:
                if (C.count_occurrences(seq, fam, k, r)
                        != C.count_by_definition(seq, fam, k, r)):
                    bad += 1
            print("  %-4s %-3d %-3d %-4d %10d | %s"
                  % (fam, k, r, n, len(paths),
                     "agrees" if bad == 0 else "%d disagree" % bad))
            rows.append([fam, k, r, n, len(paths), bad])
    write_csv("audit_paths.csv",
              ["family", "k", "r", "n", "n_paths", "disagreements"], rows)
    return rows


def audit_pmf(n=10, grid=((2, 1), (3, 2)), p2=0.4):
    banner("B. Exact PMF against brute force over 3^%d paths (p2=%.2f)" % (n, p2))
    p = C.symmetric_p(p2)
    rows = []
    print("  %-4s %-3s %-3s %16s" % ("fam", "k", "r", "max abs difference"))
    for k, r in grid:
        for fam in C.FAMILIES:
            bf = C.pmf_bruteforce(n, fam, k, r, p)
            new = C.pmf(n, fam, k, r, p)
            m = max(len(bf), len(new))

            def pad(v):
                o = np.zeros(m)
                o[:len(v)] = v
                return o
            bf, new = pad(bf), pad(new)
            d = float(np.max(np.abs(new - bf)))
            print("  %-4s %-3d %-3d %16.3e" % (fam, k, r, d))
            rows.append([fam, k, r, n, p2, d])
    write_csv("audit_pmf.csv",
              ["family", "k", "r", "n", "p2", "maxdiff"], rows)
    return rows


def main(quick=False):
    save_environment()
    grid = [(2, 1), (3, 2)] if quick else GRID
    paths = audit_paths(grid)
    pmfs = audit_pmf(n=9 if quick else 10)

    worst_paths = max(row[5] for row in paths)
    worst_pmf = max(row[5] for row in pmfs)
    banner("Summary")
    print("  disagreeing paths over the whole grid: %d" % worst_paths)
    print("  largest PMF difference against brute force: %.3e" % worst_pmf)
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    raise SystemExit(main(quick=ap.parse_args().quick))
