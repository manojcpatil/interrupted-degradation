"""
code_03_scaling.py  ->  Table 6, Figure 2
=========================================
Does the computational advantage scale with the tracking-state dimension s?
The question cannot be answered at a single setting, so this script holds n
fixed and varies (k, r) to sweep s from 3 to 46.

What the measurement shows.  Each row of A_t and B_t has at most three
non-zero entries, so nnz = O(s).  A dense implementation therefore wastes
O(s^2) work per trial and degrades as s grows, whereas a sparse implementation
costs O(s) per trial - the same order as the scalar recursion, whose order m
never exceeds s.  The s-dependent gap is dense-versus-sparse, not
scalar-versus-matrix, and the correct statement is a factor O(s) over a dense
implementation and a constant factor over a sparse one.

Outputs: Tables/tab_scaling.tex, Output/scaling.csv, Figures/fig2_scaling.png
"""

from __future__ import annotations

import argparse
import time

import numpy as np

import mcem_core as C
from common import (banner, figure_path, latex_table, save_environment,
                    write_csv, write_table)

P = (0.35, 0.35, 0.30)
GRID = [(2, 1), (3, 1), (3, 2), (4, 2), (5, 2), (6, 3), (8, 3), (10, 4)]


def _best(fn, repeats=3):
    ts = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t0)
    return float(np.min(ts))


def main(n=2000, family="E1", grid=GRID, repeats=3):
    save_environment()
    banner("Table 5 / Figure 2: cost as a function of the embedded dimension s "
           "(n=%d, %s)" % (n, family))
    rows = []
    print("  %3s %3s %5s %6s %11s %11s %13s %12s"
          % ("k", "r", "s", "nnz", "dense (s)", "sparse (s)", "dense/sparse",
             "max|dPMF|"))
    for k, r in grid:
        s = C.state_dimension(family, k, r)
        nnz = C.sparsity(family, k, r)
        td = _best(lambda: C.pmf(n, family, k, r, P), repeats)
        ts = _best(lambda: C.pmf_sparse(n, family, k, r, P), repeats)
        a = C.pmf(n, family, k, r, P)
        b = C.pmf_sparse(n, family, k, r, P)
        d = float(np.max(np.abs(a - b)))
        print("  %3d %3d %5d %6d %11.4f %11.4f %13.2f %12.2e"
              % (k, r, s, nnz, td, ts, td / ts, d))
        rows.append([family, k, r, s, nnz, n, td, ts, td / ts, d])

    write_csv("scaling.csv",
              ["family", "k", "r", "s", "nnz", "n", "dense_sec", "sparse_sec",
               "dense_over_sparse", "maxdiff"], rows)

    slope = float(np.polyfit([r_[3] for r_ in rows], [r_[4] for r_ in rows], 1)[0])
    write_table("tab_scaling.tex", latex_table(
        caption=("Cost of exact PMF evaluation as the embedded dimension $s$ "
                 "grows ($\\mathcal{E}_1$, $n=%d$, $p_0=p_1=0.35$, "
                 "$p_2=0.30$; best of %d runs)." % (n, repeats)),
        label="tab:scaling",
        colspec="rr rr rr r",
        header=[["$k$", "$r$", "$s$",
                 "$\\mathrm{nnz}(\\boldsymbol{A}_t{+}\\boldsymbol{B}_t)$",
                 "Dense (s)", "Sparse (s)", "Dense/Sparse"]],
        rows=[["%d" % r_[1], "%d" % r_[2], "%d" % r_[3], "%d" % r_[4],
               "%.4f" % r_[6], "%.4f" % r_[7], "%.2f" % r_[8]] for r_ in rows],
        notes=("The non-zero count grows linearly in $s$ (fitted slope "
               "$%.2f$), so only the dense implementation pays an $O(s^2)$ "
               "price per trial. All PMFs agree to at most $%.1e$ in absolute "
               "value." % (slope, max(r_[9] for r_ in rows)))))

    _plot(rows, n)
    return rows


def _plot(rows, n):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    s = [r[3] for r in rows]
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    ax[0].plot(s, [r[6] for r in rows], marker="o", lw=1.4,
               label="dense matrix embedding")
    ax[0].plot(s, [r[7] for r in rows], marker="s", lw=1.4,
               label="sparse matrix embedding")
    ax[0].set_xlabel("embedded dimension $s$")
    ax[0].set_ylabel("execution time (s), $n=%d$" % n)
    ax[0].set_title("Cost versus tracking-state dimension")
    ax[0].grid(alpha=0.3)
    ax[0].legend(fontsize=8)

    ax[1].plot(s, [r[4] for r in rows], marker="o", c="k", lw=1.4)
    ax[1].set_xlabel("embedded dimension $s$")
    ax[1].set_ylabel(r"$\mathrm{nnz}(\mathbf{A}_t)+\mathrm{nnz}(\mathbf{B}_t)$")
    ax[1].set_title("Sparsity of the embedding is linear in $s$")
    ax[1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(figure_path("fig2_scaling.png"), dpi=200)
    plt.close(fig)
    print("  wrote Figures/fig2_scaling.png")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--family", default="E1", choices=list(C.FAMILIES))
    a = ap.parse_args()
    main(n=a.n, family=a.family)
