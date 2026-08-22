"""
code_02_benchmark.py  ->  Table 5, Figure 1
===========================================
Covers: a like-for-like benchmark, including a sparse baseline,
and the requirement to report hardware, versions, repetitions and dispersion.

Four exact evaluations of the PMF of N_{n,2,1}^{(1)} are timed:

  dense   f_t(x) = f_{t-1}(x)A + f_{t-1}(x-1)B with dense BLAS
  sparse  the same recursion with scipy.sparse (A, B have O(s) non-zeros)
  scalar  Corollary 2, three-term scalar recursion, no matrix formed
  auto    the same recursion generated mechanically from det[I - w(A+zB)]

Every timing is the median of `--repeats` runs; the inter-quartile range is
carried into the CSV and drawn as error bars.  The maximum absolute PMF
discrepancy against the dense reference is recorded, so exactness and speed
are reported separately.

Outputs: Tables/tab_benchmark.tex, Output/benchmark.csv, Figures/fig1_benchmark.png
"""

from __future__ import annotations

import argparse
import time

import numpy as np

import mcem_core as C
from common import (banner, environment_stamp, figure_path, latex_table,
                    save_environment, sci_latex, write_csv, write_table)

P = (0.35, 0.35, 0.30)
K, R = 2, 1
DEFAULT_NS = (100, 500, 1000, 2000, 5000)


def _time(fn, n, repeats):
    ts, res = [], None
    for _ in range(repeats):
        t0 = time.perf_counter()
        res = fn(n)
        ts.append(time.perf_counter() - t0)
    ts = np.sort(np.asarray(ts))
    q1, q3 = np.percentile(ts, [25, 75])
    return float(np.median(ts)), float(q1), float(q3), res


def main(ns=DEFAULT_NS, repeats=5):
    env = save_environment()
    banner("Table 4 / Figure 1: exact PMF of $N^{(1)}_{n,2,1}$, p = %s" % (P,))
    print("  %s, Python %s, NumPy %s, SciPy %s"
          % (env["platform"], env["python"], env["numpy"], env["scipy"]))
    print("  medians of %d runs\n" % repeats)

    rec = C.scalar_recursion("E1", K, R, P)
    print("  scalar recursion: order m = %d, embedded dimension s = %d\n"
          % (rec["order"], rec["s"]))

    methods = [
        ("dense", lambda n: C.pmf(n, "E1", K, R, P)),
        ("sparse", lambda n: C.pmf_sparse(n, "E1", K, R, P)),
        ("scalar", lambda n: C.pmf_scalar_k2r1(n, P)),
        ("auto", lambda n: C.pmf_from_scalar_recursion(n, "E1", K, R, P, rec)),
    ]

    rows = []
    print("  %6s %11s %11s %11s %11s %10s %10s %13s"
          % ("n", "dense", "sparse", "scalar", "auto",
             "dense/sc", "sparse/sc", "max|dPMF|"))
    for n in ns:
        t, res = {}, {}
        for name, fn in methods:
            t[name] = _time(fn, n, repeats)
            res[name] = t[name][3]
        ref = np.asarray(res["dense"])
        diffs = {name: float(np.max(np.abs(np.asarray(v)[:len(ref)] - ref)))
                 for name, v in res.items() if name != "dense"}
        d, s_, sc, au = (t[m][0] for m in ("dense", "sparse", "scalar", "auto"))
        print("  %6d %11.5f %11.5f %11.5f %11.5f %10.2f %10.2f %13.2e"
              % (n, d, s_, sc, au, d / sc, s_ / sc, max(diffs.values())))
        rows.append([n, d, t["dense"][1], t["dense"][2],
                     s_, t["sparse"][1], t["sparse"][2],
                     sc, t["scalar"][1], t["scalar"][2], au,
                     d / sc, s_ / sc,
                     diffs["sparse"], diffs["scalar"], diffs["auto"]])

    write_csv("benchmark.csv",
              ["n", "dense_median", "dense_q1", "dense_q3",
               "sparse_median", "sparse_q1", "sparse_q3",
               "scalar_median", "scalar_q1", "scalar_q3", "auto_median",
               "ratio_dense_over_scalar", "ratio_sparse_over_scalar",
               "maxdiff_sparse", "maxdiff_scalar", "maxdiff_auto"], rows)

    write_table("tab_benchmark.tex", latex_table(
        caption=("Execution time for exact evaluation of the probability mass "
                 "function of $N^{(1)}_{n,2,1}$ with $p_0=p_1=0.35$, "
                 "$p_2=0.30$. Each entry is the median of %d independent runs; "
                 "quartiles are given in \\texttt{Output/benchmark.csv}. "
                 "Environment: %s, Python %s, NumPy %s, SciPy %s."
                 % (repeats, env["platform"], env["python"], env["numpy"],
                    env["scipy"])),
        label="tab:benchmark",
        colspec="r rrr rr c",
        header=[["$n$", "Dense (s)", "Sparse (s)", "Scalar (s)",
                 "Dense/Scalar", "Sparse/Scalar", "Max PMF difference"]],
        rows=[["%d" % r_[0], "%.4f" % r_[1], "%.4f" % r_[4], "%.4f" % r_[7],
               "%.2f" % r_[11], "%.2f" % r_[12],
               sci_latex(max(r_[13], r_[14]))] for r_ in rows],
        notes=("``Dense'' forms the full $s\\times s$ products; ``Sparse'' "
               "exploits the $O(s)$ non-zero entries; ``Scalar'' is the "
               "three-term recursion of Corollary~\\ref{cor:pmf}. Ratios below "
               "one mean the matrix implementation is the faster of the two. "
               "At $s=3$ the scalar recursion carries no asymptotic advantage; "
               "Table~\\ref{tab:scaling} varies $s$.")))

    _plot(rows, repeats)
    return rows


def _plot(rows, repeats):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ns = [r[0] for r in rows]
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    for lab, c, q1, q3, mk in [("Dense matrix embedding", 1, 2, 3, "o"),
                               ("Sparse matrix embedding", 4, 5, 6, "s"),
                               ("Scalar recursion (Cor. 2)", 7, 8, 9, "^")]:
        med = [r[c] for r in rows]
        lo = [max(r[c] - r[q1], 0) for r in rows]
        hi = [max(r[q3] - r[c], 0) for r in rows]
        ax[0].errorbar(ns, med, yerr=[lo, hi], marker=mk, capsize=3, lw=1.4,
                       label=lab)
    ax[0].set_xscale("log")
    ax[0].set_yscale("log")
    ax[0].set_xlabel("sequence length $n$")
    ax[0].set_ylabel("execution time (s), median of %d runs" % repeats)
    ax[0].set_title("Exact PMF evaluation, $k=2$, $r=1$ ($s=3$)")
    ax[0].grid(True, which="both", alpha=0.3)
    ax[0].legend(fontsize=8)

    ax[1].plot(ns, [r[11] for r in rows], marker="o", label="dense / scalar")
    ax[1].plot(ns, [r[12] for r in rows], marker="s", label="sparse / scalar")
    ax[1].axhline(1.0, ls="--", c="grey", lw=1)
    ax[1].set_xscale("log")
    ax[1].set_xlabel("sequence length $n$")
    ax[1].set_ylabel("time ratio")
    ax[1].set_title("Relative cost (values $<1$ favour the matrix method)")
    ax[1].grid(True, alpha=0.3)
    ax[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(figure_path("fig1_benchmark.png"), dpi=200)
    plt.close(fig)
    print("  wrote Figures/fig1_benchmark.png")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--ns", type=int, nargs="*", default=list(DEFAULT_NS))
    a = ap.parse_args()
    main(ns=a.ns, repeats=a.repeats)
