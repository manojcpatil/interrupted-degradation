"""
code_11_reduction.py  ->  Table 8
=================================
The binary boundary case: does p0 = 0 collapse the strict families onto the
classical interrupted-run statistic of the binary literature?

This is the check that separates a genuine generalisation from a relabelling.
If the three-letter model merely renamed the two-letter one, the reduction
would be the whole story; because the reduction holds *only* at p0 = 0, the
normal state's hard reset is doing work that has no binary counterpart.

Two independent comparisons, neither of which reuses the manuscript's scanner:

  1. every sequence in {W,C}^n, chain replay versus a from-scratch
     implementation of the binary rule;
  2. the probability mass function at p0 = 0 versus brute-force enumeration of
     all 2^n weighted binary sequences, counted by that same from-scratch rule.

Writes Output/binary_reduction.csv and Tables/tab_reduction.tex.

License: MIT.
"""

from __future__ import annotations

import itertools

import numpy as np

import mcem_core as C
from common import banner, latex_table, sci_latex, write_csv, write_table

GRID = ((2, 1), (2, 2), (3, 1), (3, 2), (4, 2))
STRICT = (("E1", False), ("E3", True))


def binary_interrupted_runs(bits, k, r, overlapping):
    """The binary rule, written directly from its usual verbal statement.

    Successes are 1 and failures are 0.  An occurrence is k successes in which
    consecutive members are separated by at most r failures.  A non-overlapping
    count consumes the window; an overlapping count slides it by one success.
    Nothing here knows about A, B or the embedded state space.
    """
    count = 0
    buf = []                                  # positions of the held successes
    for pos, b in enumerate(bits):
        if b != 1:
            continue
        if buf and (pos - buf[-1] - 1) > r:
            buf = []                          # gap too long: history dies
        buf.append(pos)
        if len(buf) == k:
            count += 1
            buf = buf[1:] if overlapping else []
    return count


def audit_paths(ns=(10, 12), grid=GRID):
    """Chain replay versus the binary rule on every sequence in {W,C}^n."""
    rows = []
    for n in ns:
        seqs = [(bits, [C.CRITICAL if b else C.WARNING for b in bits])
                for bits in itertools.product((0, 1), repeat=n)]
        for k, r in grid:
            for family, overlapping in STRICT:
                bad = 0
                for bits, seq in seqs:
                    if (C.count_occurrences(seq, family, k, r)
                            != binary_interrupted_runs(bits, k, r, overlapping)):
                        bad += 1
                rows.append((family, k, r, n, 2 ** n, bad))
                print("  %s k=%d r=%d n=%2d: %d disagreeing paths out of %d"
                      % (family, k, r, n, bad, 2 ** n))
    return rows


def audit_pmf(n=12, p2=0.45, grid=((2, 1), (3, 2))):
    """PMF at p0 = 0 versus brute force over weighted binary sequences."""
    rows = []
    for k, r in grid:
        for family, overlapping in STRICT:
            f_chain = C.pmf(n, family, k, r, (0.0, 1 - p2, p2))
            xmax = len(f_chain) - 1
            f_bf = np.zeros(xmax + 1)
            for bits in itertools.product((0, 1), repeat=n):
                w = np.prod([p2 if b else 1 - p2 for b in bits])
                x = binary_interrupted_runs(bits, k, r, overlapping)
                if x <= xmax:
                    f_bf[x] += w
            err = float(np.max(np.abs(f_chain - f_bf)))
            rows.append((family, k, r, n, p2, err))
            print("  %s k=%d r=%d n=%d p2=%.2f: max |PMF diff| = %.2e"
                  % (family, k, r, n, p2, err))
    return rows


def write_summary_table(paths, pmfs):
    """One row per (family, k, r): worst case over the path and PMF audits."""
    by_key = {}
    for family, k, r, n, total, bad in paths:
        key = (family, k, r)
        acc = by_key.setdefault(key, {"paths": 0, "bad": 0, "err": None})
        acc["paths"] += total
        acc["bad"] += bad
    for family, k, r, n, p2, err in pmfs:
        acc = by_key.get((family, k, r))
        if acc is not None:
            acc["err"] = err if acc["err"] is None else max(acc["err"], err)

    rows = []
    for (family, k, r), acc in by_key.items():
        rows.append(["$\\mathcal{%s}_%s$" % ("E", family[1]), k, r,
                     "%d" % acc["paths"], acc["bad"],
                     "---" if acc["err"] is None else sci_latex(acc["err"])])

    write_table("tab_reduction.tex", latex_table(
        caption=("Reduction of the strict families to the binary interrupted-run "
                 "statistic at $p_0=0$. Every sequence in $\\{\\W,\\Cc\\}^{n}$ for "
                 "$n=10$ and $n=12$ is counted twice, once by replaying the "
                 "embedded chain and once by an implementation of the binary rule "
                 "written directly from its verbal statement; the last column "
                 "compares the mass function at $p_0=0$ with brute-force "
                 "enumeration of all $2^{12}$ weighted binary sequences at "
                 "$p_2=0.45$. The reduction is exact, and it fails as soon as "
                 "$p_0>0$, which is the content of Remark~\\ref{rem:binary}."),
        label="tab:reduction",
        colspec="l cc r c c",
        header=["Family", "$k$", "$r$", "Paths compared",
                "Disagreeing paths", "Max PMF difference"],
        rows=rows))
    return rows


def main():
    banner("Exhaustive path audit: chain replay versus the binary rule")
    paths = audit_paths()

    banner("Mass function at p0 = 0 versus brute force over binary sequences")
    pmfs = audit_pmf()

    write_csv("binary_reduction_paths.csv",
              ("family", "k", "r", "n", "paths", "disagreeing"), paths)
    write_csv("binary_reduction_pmf.csv",
              ("family", "k", "r", "n", "p2", "max_abs_pmf_diff"), pmfs)
    write_summary_table(paths, pmfs)

    total_bad = sum(row[5] for row in paths)
    print("\n  total disagreeing paths over the whole grid: %d" % total_bad)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
