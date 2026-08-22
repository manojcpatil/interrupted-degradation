"""
code_10_figures.py  ->  Figures 3 and 4
=======================================
Two things this script is careful about.  The horizon is a command-line
argument and is written into both the file name and the figure title, so the
caption, the file name and the surrounding tables cannot drift apart.  And the
stochastic ordering is a partial order, not a hierarchy, because two of the
CDFs cross.  The dominance
check is computed, not asserted, and its verdict is written to
Tables/stochastic_order.tex for the manuscript to quote.

Outputs: Figures/fig3_sensitivity_n<N>.png, Figures/fig4_cdf_n<N>.png,
         Tables/stochastic_order.tex, Output/stochastic_order.csv
"""

from __future__ import annotations

import argparse

import numpy as np

import mcem_core as C
from common import (banner, figure_path, save_environment, write_csv,
                    write_table)

LABEL = {"E1": r"$\mathcal{E}_1$ strict, non-overlapping",
         "E2": r"$\mathcal{E}_2$ threshold, non-overlapping",
         "E3": r"$\mathcal{E}_3$ strict, overlapping",
         "E4": r"$\mathcal{E}_4$ threshold, overlapping"}


def sensitivity_grid(n, k, r, p2s):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(11, 7.4), constrained_layout=True)
    for ax, fam in zip(axes.ravel(), C.FAMILIES):
        for p2 in p2s:
            f = C.pmf(n, fam, k, r, C.symmetric_p(p2))
            keep = np.nonzero(f > 1e-6)[0]
            hi = int(keep.max()) if keep.size else 0
            ax.plot(np.arange(hi + 1), f[:hi + 1], marker="o", ms=3, lw=1.3,
                    label=r"$p_2=%.1f$" % p2)
        ax.set_title(LABEL[fam], fontsize=10)
        ax.set_xlabel("$x$")
        ax.set_ylabel(r"$\Pr(N=x)$")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7)
    fig.suptitle("Exact probability mass functions across degradation "
                 "intensities ($n=%d$, $k=%d$, $r=%d$)" % (n, k, r), fontsize=11)
    path = figure_path("fig3_sensitivity_n%d.png" % n)
    fig.savefig(path, dpi=200)
    plt.close(fig)
    print("  wrote Figures/fig3_sensitivity_n%d.png" % n)


def cdf_comparison(n, k, r, p2):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8.6, 5.0))
    stats = []
    for fam in C.FAMILIES:
        f = C.pmf(n, fam, k, r, C.symmetric_p(p2))
        c = np.cumsum(f)
        hi = min(int(np.searchsorted(c, 1 - 1e-9)) + 1, len(c) - 1)
        ax.step(np.arange(hi + 1), c[:hi + 1], where="post", lw=1.7,
                label=LABEL[fam])
        mu, var = C.pmf_moments(f)
        stats.append((fam, mu, var))
    ax.set_xlabel("$x$")
    ax.set_ylabel(r"$\Pr(N \leq x)$")
    ax.set_title("Comparative cumulative distribution functions "
                 "($n=%d$, $k=%d$, $r=%d$, $p_2=%.1f$)" % (n, k, r, p2))
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9)
    fig.tight_layout()
    path = figure_path("fig4_cdf_n%d.png" % n)
    fig.savefig(path, dpi=200)
    plt.close(fig)
    print("  wrote Figures/fig4_cdf_n%d.png" % n)
    print("  %-4s %12s %12s" % ("fam", "mean", "variance"))
    for fam, mu, var in stats:
        print("  %-4s %12.6f %12.6f" % (fam, mu, var))
    return stats


def dominance(n, k, r, p2):
    """Pairwise first-order stochastic dominance, computed not asserted."""
    cdfs, L = {}, 0
    for fam in C.FAMILIES:
        f = C.pmf(n, fam, k, r, C.symmetric_p(p2))
        cdfs[fam] = np.cumsum(f)
        L = max(L, len(f))
    for fam in cdfs:
        pad = np.ones(L)
        pad[:len(cdfs[fam])] = cdfs[fam]
        cdfs[fam] = pad

    rows, ordered, crossing = [], [], []
    for i, a in enumerate(C.FAMILIES):
        for b in C.FAMILIES[i + 1:]:
            ge = bool(np.all(cdfs[a] >= cdfs[b] - 1e-12))
            le = bool(np.all(cdfs[a] <= cdfs[b] + 1e-12))
            if ge:
                verdict = "%s <=st %s" % (a, b)
                ordered.append((a, b))
            elif le:
                verdict = "%s <=st %s" % (b, a)
                ordered.append((b, a))
            else:
                verdict = "not ordered (CDFs cross)"
                crossing.append((a, b))
            rows.append([a, b, verdict])
            print("    %s vs %s : %s" % (a, b, verdict))
    write_csv("stochastic_order.csv", ["family_a", "family_b", "verdict"], rows)

    M = {"E1": "\\mathcal{E}_1", "E2": "\\mathcal{E}_2",
         "E3": "\\mathcal{E}_3", "E4": "\\mathcal{E}_4"}
    chain = ", ".join("$N^{(%s)} \\preceq_{\\mathrm{st}} N^{(%s)}$"
                      % (a[1], b[1]) for a, b in ordered)
    if crossing:
        word = "pair" if len(crossing) == 1 else "pairs"
        verb = "is" if len(crossing) == 1 else "are"
        cross = ("; the %s %s %s not comparable, the distribution functions "
                 "crossing" % (word,
                               ", ".join("$(%s,%s)$" % (M[a], M[b])
                                         for a, b in crossing), verb))
    else:
        cross = "; every pair is comparable"
    text = ("%% generated by code_10_figures.py, n=%d k=%d r=%d p2=%.2f\n"
            "At these parameters the pairwise comparisons are %s%s.\n"
            % (n, k, r, p2, chain, cross))
    write_table("stochastic_order.tex", text)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--r", type=int, default=2)
    ap.add_argument("--p2-grid", type=float, nargs="*",
                    default=[0.3, 0.5, 0.7, 0.9])
    ap.add_argument("--p2-cdf", type=float, default=0.6)
    a = ap.parse_args()
    save_environment()
    banner("Figures 3 and 4 (n=%d, k=%d, r=%d)" % (a.n, a.k, a.r))
    sensitivity_grid(a.n, a.k, a.r, a.p2_grid)
    cdf_comparison(a.n, a.k, a.r, a.p2_cdf)
    print("\n  first-order stochastic dominance at p2 = %.2f:" % a.p2_cdf)
    dominance(a.n, a.k, a.r, a.p2_cdf)


if __name__ == "__main__":
    main()
